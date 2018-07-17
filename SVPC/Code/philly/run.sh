#! /bin/bash
set -x

python3 $1.py --model_type k --full_connect_hn 256,128,64,32 --stacking False --nfold 5 \
        --batch_size 16 --full_connect_dropout 0.1 \
        --epochs 50 --ensemble_nfold 5 --batch_interval 20000 \
        --input-training-data-path ../../Data/ --output-model-path ../../Data/ \
        --debug False --neg_sample False --sample_C 40 --load_only_singleCnt True --log_transform False \
        --search_best_iteration False --best_iteration 1000 --search_iterations 600,1300,100 \
        --input-previous-model-path ../Data/lgb_models/philly/ --split_train_val False --train_eval_len 110000000 \
        --eval_len 10000000 --test_for_train False --blend_tune False --stacking False --lgb_boost_dnn false \
        --lgb_ensemble_nfold 5 --load_from_pickle False

# python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type vdcnn \
#         --max_seq_len 100 --nfold 2 --emb_dim 32 --epochs 2 --batch_size 128 --ensemble_nfold 1 \
#         --vdcnn_filters "4, 8" --full_connect_dropout 0 --vdcc_top_k 8 \
#         --full_connect_hn "4, 0" --char_split True --load_wv_model False