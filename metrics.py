import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer
from datasets import Dataset
from accelerate import Accelerator
from peft import PeftModel
from scipy.stats import spearmanr, pearsonr
from sklearn import metrics

# Helper functions and data preparation
def truncate_labels(labels, max_length):
	"""Truncate labels to the specified max_length."""
	return [label[:max_length] for label in labels]

def compute_metrics(p):
	"""Compute metrics for evaluation."""
	predictions, labels = p
	predictions = np.argmax(predictions, axis=2)

	# Remove padding (-100 labels)
	predictions = predictions[labels != -100].flatten()
	labels = labels[labels != -100].flatten()

	# Compute accuracy
	accuracy = accuracy_score(labels, predictions)

	# Compute precision, recall, F1 score, and AUC
	precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
	auc = roc_auc_score(labels, predictions)

	# Compute MCC
	mcc = matthews_corrcoef(labels, predictions)

	return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc}

# torchdrug
def area_under_roc(pred, target):
	"""
	Area under receiver operating characteristic curve (ROC).

	Parameters:
		pred (Tensor): predictions of shape :math:`(n,)`
		target (Tensor): binary targets of shape :math:`(n,)`
	"""
	order = pred.argsort(descending=True)
	target = target[order]
	hit = target.cumsum(0)
	all = (target == 0).sum() * (target == 1).sum()
	auroc = hit[target == 0].sum() / (all + 1e-10)
	return auroc

def area_under_prc(pred, target):
	"""
	Area under precision-recall curve (PRC).

	Parameters:
		pred (Tensor): predictions of shape :math:`(n,)`
		target (Tensor): binary targets of shape :math:`(n,)`
	"""
	order = pred.argsort(descending=True)
	target = target[order]
	precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
	auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
	return auprc

# torchdrug
def f1_max(pred, target):
	"""
	F1 score with the optimal threshold.

	This function first enumerates all possible thresholds for deciding positive and negative
	samples, and then pick the threshold with the maximal F1 score.

	Parameters:
		pred (Tensor): predictions of shape :math:`(B, N)`
		target (Tensor): binary targets of shape :math:`(B, N)`
	"""
	order = pred.argsort(descending=True, dim=1)
	target = target.gather(1, order)
	precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
	recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
	is_start = torch.zeros_like(target).bool()
	is_start[:, 0] = 1
	is_start = torch.scatter(is_start, 1, order, is_start)

	all_order = pred.flatten().argsort(descending=True)
	order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
	order = order.flatten()
	inv_order = torch.zeros_like(order)
	inv_order[order] = torch.arange(order.shape[0], device=order.device)
	is_start = is_start.flatten()[all_order]
	all_order = inv_order[all_order]
	precision = precision.flatten()
	recall = recall.flatten()
	all_precision = precision[all_order] - \
					torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
	all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
	all_recall = recall[all_order] - \
				 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
	all_recall = all_recall.cumsum(0) / pred.shape[0]
	all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
	return all_f1.max()

def accuracy(pred, target):
	return (pred.argmax(dim=-1) == target).float().mean()

def compute_cls_metrics(p):
	# preds: (num_samples, num_cls), fp32
	# labels: (num_samples, num_cls), fp32
	preds, labels = p
	#preds = preds.reshape(-1)
	#labels = labels.reshape(-1)
	preds = torch.tensor(preds)
	labels = torch.tensor(labels)
	auprc_micro = area_under_prc(preds.flatten(), labels.long().flatten())
	f1max = f1_max(preds, labels)
	return {'auprc@micro': auprc_micro, 'f1_max': f1max}


""" old 
class ComputeClsMetrics:
	def __init__(self):
		self.batch_auprc_micro = []
		self.batch_f1max = []
	
	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		batch_size = len(preds)
		auprc_micro = area_under_prc(preds.flatten(), labels.long().flatten())
		f1max = f1_max(preds, labels)
		self.batch_auprc_micro.extend([auprc_micro.cpu()] * batch_size)
		self.batch_f1max.extend([f1max.cpu()] * batch_size)
		if compute_result:
			result = {'auprc@micro': np.mean(self.batch_auprc_micro).item(), 'f1_max': np.mean(self.batch_f1max).item()}
			self.batch_auprc_micro = []
			self.batch_f1max = []
			return result
"""

class ComputeClsMetrics:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		self.batch_preds.extend(preds.cpu())
		self.batch_labels.extend(labels.cpu())

		if compute_result:
			result = {
				'auprc@micro': area_under_prc(torch.stack(self.batch_preds).flatten(), torch.stack(self.batch_labels).long().flatten()),
				'f1_max': f1_max(torch.stack(self.batch_preds), torch.stack(self.batch_labels))}
			self.batch_preds, self.batch_labels = [], []
			return result

class ComputeMultiClsMetrics:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		self.batch_preds.extend(preds.cpu())
		self.batch_labels.extend(labels.cpu())

		if compute_result:
			result = {
				'accuracy': accuracy(torch.stack(self.batch_preds), torch.stack(self.batch_labels).long())}
			self.batch_preds, self.batch_labels = [], []
			return result

def compute_reg_metrics(p):
	predictions, labels = p
	predictions = predictions.reshape(-1)
	labels = labels.reshape(-1)
	spearman = spearmanr(predictions, labels)[0]
	pearson = pearsonr(predictions, labels)[0]
	rmse = np.sqrt(metrics.mean_squared_error(predictions, labels))
	r2 = metrics.r2_score(labels, predictions)
	return {'spearman': spearman, 'pearson': pearson, 'rmse': rmse, 'r2': r2}

class ComputeTokenClsMetrics:
	def __init__(self):
		self.batch_preds, self.batch_labels = [], []

	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		# preds = preds[:, 1:-1]
		# preds = np.argmax(preds, axis=2)
		batch_size, seq_len, num_cls = preds.shape
		assert batch_size == 1
		out_label_list = [[] for _ in range(batch_size)]
		preds_list = [[] for _ in range(batch_size)]
		for i in range(batch_size):
			for j in range(seq_len):
				if labels[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
					out_label_list[i].append(labels[i][j].detach().cpu())
					preds_list[i].append(preds[i][j].detach().cpu())
		self.batch_preds.extend(preds_list[0])
		self.batch_labels.extend(out_label_list[0])
		if compute_result:
			result = {
				'accuracy': accuracy(torch.stack(self.batch_preds), torch.stack(self.batch_labels).long()),
			}
			self.batch_preds, self.batch_labels = [], []
			return result

""" old 
class ComputeRegMetrics:
	def __init__(self):
		self.batch_spearman = []
		self.batch_pearson = []
		self.batch_rmse = []
		self.batch_r2 = []
	
	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		batch_size = len(preds)
		preds = preds.reshape(-1).cpu().numpy()
		labels = labels.reshape(-1).cpu().numpy()
		spearman = spearmanr(preds, labels)[0]
		pearson = pearsonr(preds, labels)[0]
		rmse = np.sqrt(metrics.mean_squared_error(preds, labels))
		r2 = metrics.r2_score(preds, labels)
		self.batch_spearman.extend([spearman] * batch_size)
		self.batch_pearson.extend([pearson] * batch_size)
		self.batch_rmse.extend([rmse] * batch_size)
		self.batch_r2.extend([r2] * batch_size)        
		if compute_result:
			result = {'spearman': np.mean(self.batch_spearman).item(),
					  'pearson': np.mean(self.batch_pearson).item(),
					  'rmse': np.mean(self.batch_rmse).item(),
					  'r2': np.mean(self.batch_r2).item(),}
			self.batch_auprc_micro = []
			self.batch_f1max = []
			self.batch_rmse = []
			self.batch_r2 = []
			return result
"""


class ComputeRegMetrics:
	def __init__(self):
		self.batch_preds = []
		self.batch_labels = []
	
	def __call__(self, p, compute_result):
		preds, labels = p
		if isinstance(preds, tuple):
			preds = preds[0]
		if not isinstance(preds, torch.Tensor):
			preds = torch.tensor(preds)
		if not isinstance(labels, torch.Tensor):
			labels = torch.tensor(labels)
		preds = preds.reshape(-1).cpu().numpy()
		labels = labels.reshape(-1).cpu().numpy()

		self.batch_preds.extend(preds)
		self.batch_labels.extend(labels)
     
		if compute_result:
			result = {'spearman': spearmanr(self.batch_preds, self.batch_labels)[0],
					  'pearson': pearsonr(self.batch_preds, self.batch_labels)[0],
					  'rmse': np.sqrt(metrics.mean_squared_error(self.batch_preds, self.batch_labels)),
					  'r2': metrics.r2_score(self.batch_labels, self.batch_preds)}
			self.batch_preds = []
			self.batch_labels = []
			return result
		

class WeightedTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		"""Custom compute_loss function."""
		outputs = model(**inputs)
		loss_fct = nn.CrossEntropyLoss()
		active_loss = inputs["attention_mask"].view(-1) == 1
		active_logits = outputs.logits.view(-1, model.config.num_labels)
		active_labels = torch.where(
			active_loss, inputs["labels"].view(-1), torch.tensor(loss_fct.ignore_index).type_as(inputs["labels"])
		)
		loss = loss_fct(active_logits, active_labels)
		return (loss, outputs) if return_outputs else loss

