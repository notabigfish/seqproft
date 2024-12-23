export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/Fold-nf-mha.yml 
python train_search.py configs/Fold-f-mha.yml 

python train_search.py configs/Fold-nf-smh.yml 
python train_search.py configs/Fold-f-smh.yml 

python train_search.py configs/Fold-nf-smha.yml 
python train_search.py configs/Fold-f-smha.yml

python train_search.py configs/Fold-nf-mha-35M.yml
python train_search.py configs/Fold-f-mha-35M.yml 
python train_search.py configs/Fold-nf-mha-150M.yml
python train_search.py configs/Fold-f-mha-150M.yml 

python train_search.py configs/Fold-nf-smha-35M.yml
python train_search.py configs/Fold-f-smha-35M.yml 
python train_search.py configs/Fold-nf-smha-150M.yml
python train_search.py configs/Fold-f-smha-150M.yml 
