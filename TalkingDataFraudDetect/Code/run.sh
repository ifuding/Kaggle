#! /bin/bash
set -x

python3 $1.py --model_type l --full_connect_hn 2 --stacking False --nfold 5 \
        --batch_size 1024 --full_connect_dropout 0 \
        --epochs 4 --ensemble_nfold 1 --batch_interval 20000 \
        --input-training-data-path ../Data/123Cnt_DayByDay_AttrOverCnt_KeyCount/AttrOverCntAndTrainValCountVar/  \
        --output-model-path ../Data/ --emb_dim "5,5,5,5,5,5,4" \
        --debug False --neg_sample False --sample_C 1 --load_only_singleCnt True --log_transform False \
        --search_best_iteration True --best_iteration 600

# python3 PoolGRU.py --input-training-data-path ../../Data/ --output-model-path . --model_type vdcnn \
#         --max_seq_len 100 --nfold 2 --emb_dim 32 --epochs 2 --batch_size 128 --ensemble_nfold 1 \
#         --vdcnn_filters "4, 8" --full_connect_dropout 0 --vdcc_top_k 8 \
#         --full_connect_hn "4, 0" --char_split True --load_wv_model False