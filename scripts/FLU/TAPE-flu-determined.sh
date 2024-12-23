export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/FLU-nf-mha.yml 
python train_search.py configs/FLU-f-mha.yml 

python train_search.py configs/FLU-nf-smh.yml 
python train_search.py configs/FLU-f-smh.yml 

python train_search.py configs/FLU-nf-smha.yml
python train_search.py configs/FLU-f-smha.yml

python train_search.py configs/FLU-nf-mha-35M.yml
python train_search.py configs/FLU-f-mha-35M.yml
python train_search.py configs/FLU-nf-mha-150M.yml
python train_search.py configs/FLU-f-mha-150M.yml

python train_search.py configs/FLU-nf-smha-35M.yml
python train_search.py configs/FLU-f-smha-35M.yml
python train_search.py configs/FLU-nf-smha-150M.yml
python train_search.py configs/FLU-f-smha-150M.yml
