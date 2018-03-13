#! /bin/bash

# python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type cnn \
#         --max_seq_len 10 --nfold 2 --emb_dim 300 --epochs 2 --batch_size 32 --ensemble_nfold 2 \
#         --filter_size 16 --batch_interval 1000 --full_connect_dropout 0 --emb_dropout 0 \
#         --full_connect_hn "16, 8" --rnn_unit 4 --vocab_size 1000

python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type vdcnn \
        --max_seq_len 10 --nfold 2 --emb_dim 32 --epochs 2 --batch_size 32 --ensemble_nfold 2 \
        --vdcnn_filters "16, 32" --full_connect_dropout 0 --vdcc_top_k 8 \
        --full_connect_hn "16, 8" --char_split True --load_wv_model False