#! /bin/bash

python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --vocab_size 1000 \
        --max_seq_len 10 --nfold 2 --emb_dim 300 --epochs 2 --batch_size 32 --ensemble_nfold 2 \
        --filter_size 16 --batch_interval 1000 --full_connect_dropout 0 --emb_dropout 0 \
        --full_connect_hn 0 --rnn_unit 16