CUDA_VISIBLE_DEVICES=0 python3 inference_postfilter.py \
            --dataset StreetviewFilter \
            --max_desc_length 256 \
            --max_question_length 256 \
            --batch_size 1 \
            --save_dir ./checkpoints/FDT5/checkpoint-[FILL YOURS] \
            --save_name FDT5 \
            --remain_samples 1 \
            --num_samples 1
