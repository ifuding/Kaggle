#! /bin/bash

python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type l \
        --max_seq_len 100 --nfold 10 --emb_dim 300 --epochs 1 --batch_size 64 --ensemble_nfold 3 \
        --filter_size 128 --batch_interval 1000000 --full_connect_dropout 0 --emb_dropout 0 \
        --full_connect_hn 32 --rnn_unit 128 --vocab_size 150000 --separate_label_layer True --stem False \
        --wv_model_file wiki.en.vec.indata -resnet_hn False --vdcc_top_k 1 --char_split False --load_wv_model True \
        --fix_wv_model False --kernel_size_list 1,2,3,4,5,6,7 --rnn_input_dropout 0 --rnn_state_dropout 0 \
        --stacking False --uniform_init_emb False --load_stacking_data True

# python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type vdcnn \
#         --max_seq_len 100 --nfold 2 --emb_dim 32 --epochs 2 --batch_size 128 --ensemble_nfold 1 \
#         --vdcnn_filters "4, 8" --full_connect_dropout 0 --vdcc_top_k 8 \
#         --full_connect_hn "4, 0" --char_split True --load_wv_model False