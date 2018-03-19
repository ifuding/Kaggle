#! /bin/bash

python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type cnn \
        --max_seq_len 100 --nfold 2 --emb_dim 300 --epochs 1 --batch_size 2048 --ensemble_nfold 2 \
        --filter_size 2 --batch_interval 1000000 --full_connect_dropout 0 --emb_dropout 0 \
        --full_connect_hn 2 --rnn_unit 2 --vocab_size 300000 --separate_label_layer False --stem False \
        --wv_model_file wiki.en.vec.indata -resnet_hn False --vdcc_top_k 1 --char_split False --load_wv_model False \
        --fix_wv_model True --kernel_size_list 1,2,3,4,5,6,7 --rnn_input_dropout 0 --rnn_state_dropout 0 \
        --stacking True --uniform_init_emb False

# python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type vdcnn \
#         --max_seq_len 100 --nfold 2 --emb_dim 32 --epochs 2 --batch_size 128 --ensemble_nfold 1 \
#         --vdcnn_filters "4, 8" --full_connect_dropout 0 --vdcc_top_k 8 \
#         --full_connect_hn "4, 0" --char_split True --load_wv_model False