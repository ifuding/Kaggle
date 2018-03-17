#! /bin/bash

python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type cnn \
        --max_seq_len 2048 --nfold 10 --emb_dim 300 --epochs 2 --batch_size 256 --ensemble_nfold 2 \
        --filter_size 16,0 --batch_interval 1000000 --full_connect_dropout 0 --emb_dropout 0 \
        --full_connect_hn 0,0 --rnn_unit 0 --vocab_size 70 --separate_label_layer False --stem False \
        --wv_model_file wiki.en.vec.indata -resnet_hn False --vdcc_top_k 1 --char_split True --load_wv_model True \
        --fix_wv_model False --letter_num 3

# python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type vdcnn \
#         --max_seq_len 100 --nfold 2 --emb_dim 32 --epochs 2 --batch_size 128 --ensemble_nfold 1 \
#         --vdcnn_filters "4, 8" --full_connect_dropout 0 --vdcc_top_k 8 \
#         --full_connect_hn "4, 0" --char_split True --load_wv_model False