#! /bin/bash

python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --vocab_size 1000 --max_seq_len 10 --nfold 2 --emb_dim 300 --epochs 2 --batch_size 1024 --ensemble_nfold 2 --filter_size 16