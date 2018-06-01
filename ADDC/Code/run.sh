#! /bin/bash
set -x

python3 $1.py --model_type l --full_connect_hn 8 --stacking False --nfold 5 \
        --batch_size 1024 --full_connect_dropout 0 \
        --epochs 4 --ensemble_nfold 5 --batch_interval 20000 \
        --input-training-data-path ../Data/  \
        --output-model-path ../Data/ --emb_dim 5,5,5,5,5,5,5,5,5,5,5,5,5,5 \
        --debug False --neg_sample False --sample_C 40 --load_only_singleCnt True --log_transform False \
        --search_best_iteration False --best_iteration 1000 --search_iterations 600,1300,100 \
        --input-previous-model-path ../Data/ --split_train_val False --train_eval_len 110000000 \
        --eval_len 10000000 --test_for_train False --blend_tune False --stacking False --vocab_size 300000 \
        --max_len 200,20 --load_wv_model True --wv_model_file cc.ru.300.vec \
        --kernel_size_list 1,2,3 --filter_size 8 --rnn_units 0 --fix_wv_model True --lgb_boost_dnn False \
        --lgb_ensemble_nfold 5

# python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type vdcnn \
#         --max_seq_len 100 --nfold 2 --emb_dim 32 --epochs 2 --batch_size 128 --ensemble_nfold 1 \
#         --vdcnn_filters "4, 8" --full_connect_dropout 0 --vdcc_top_k 8 \
#         --full_connect_hn "4, 0" --char_split True --load_wv_model False