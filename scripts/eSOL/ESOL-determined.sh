export PYTHONPATH=$PWD:$PYTHONPATH
export TRANSFORMERS_CACHE=$PWD/dwnl_ckpts/huggingface_cache
python train_search.py configs/eSOL/ESOL-nf-mha.yml 
python train_search.py configs/eSOL/ESOL-f-mha.yml 

python train_search.py configs/eSOL/ESOL-nf-smh.yml 
python train_search.py configs/eSOL/ESOL-f-smh.yml 

python train_search.py configs/eSOL/ESOL-nf-smha.yml 
python train_search.py configs/eSOL/ESOL-f-smha.yml

python train_search.py configs/eSOL/ESOL-nf-mha-35M.yml 
python train_search.py configs/eSOL/ESOL-f-mha-35M.yml 
python train_search.py configs/eSOL/ESOL-nf-mha-150M.yml
python train_search.py configs/eSOL/ESOL-f-mha-150M.yml 

python train_search.py configs/eSOL/ESOL-nf-smha-35M.yml 
python train_search.py configs/eSOL/ESOL-f-smha-35M.yml 
python train_search.py configs/eSOL/ESOL-nf-smha-150M.yml
python train_search.py configs/eSOL/ESOL-f-smha-150M.yml 

