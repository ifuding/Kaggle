#! /bin/bash

python3 main.py --model_type k --full_connect_hn 4 --stacking False --nfold 5 \
        --batch_size 1024 --full_connect_dropout 0 \
        --epochs 1 --ensemble_nfold 1 --batch_interval 20000 \
        --input-training-data-path ../Data/ --output-model-path . --emb_dim "5,5,5,5,5,5,4" \
        --debug False

# python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type vdcnn \
#         --max_seq_len 100 --nfold 2 --emb_dim 32 --epochs 2 --batch_size 128 --ensemble_nfold 1 \
#         --vdcnn_filters "4, 8" --full_connect_dropout 0 --vdcc_top_k 8 \
#         --full_connect_hn "4, 0" --char_split True --load_wv_model False