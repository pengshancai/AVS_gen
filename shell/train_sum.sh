CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --model_name_or_path allenai/led-base-16384 \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../AVS_gen_/data/sum/train.json \
    --validation_file ../AVS_gen_/data/sum/valid.json \
    --test_file ../AVS_gen_/data/sum/test.json \
    --output_dir ../AVS_gen_/dump/mimic-led \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --num_train_epochs 6 \
    --save_steps 3000 \
    --save_total_limit 3 \
    --max_source_length 1024 \
    --max_target_length 512

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --model_name_or_path allenai/led-base-16384 \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ../avs/data/json/src-tgt_1.0/train.json \
    --validation_file ../avs/data/json/src-tgt_1.0/val.json \
    --test_file ../avs/data/json/src-tgt_1.0/test.json \
    --output_dir ../AVS_gen_/dump/mimic-led \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --num_train_epochs 6 \
    --save_steps 3000 \
    --save_total_limit 3 \
    --max_source_length 1024 \
    --max_target_length 512

