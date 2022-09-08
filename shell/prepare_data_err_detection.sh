# This script will split the entire training set into 5 folds.
# For each fold, the script make a language modeling dataset by adding some noise
# Please revise the value of the following hyper-parameters: input_txt_dir, input_json_dir, output_dir
sh scripts/data_process/err_det-masked_events.py \
    --input_txt_dir path/to/tgt_txt/ \
    --input_json_dir path/to/tgt_ent/ \
    --output_dir path/to/your_ml_data_dir \
    --num_folds 5

# The following command will train 5 BART language models
# Each model is trained on four folds of the data
# Please revise the value of the following hyper-parameters:
#   data_dir, output_dir
data_dir = path/to/your_ml_data_dir/
for FOLD in 0 1 2 3 4
do
    python ./run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --train_file "${data_dir}fold_${FOLD}/train.json" \
    --validation_file "${data_dir}fold_${FOLD}/inference.json" \
    --output_dir path/to/your_ml_model_dump_dir \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --num_train_epochs 3 \
    --save_steps 9000 \
    --save_total_limit 3 \
    --max_source_length 256 \
    --max_target_length 256 \
    --text_column text \
    --summary_column summary
done

# The following command will use the language model you have just trained to generate the train/valid set for error detection model
sh scripts/data_process/err_det-build_datasets.py \
    --input_src_txt_dir path/to/src_txt/
    --input_tgt_txt_dir path/to/tgt_txt/ \
    --input_tgt_json_dir path/to/tgt_ent/ \
    --output_dir path/to/your_err_det_data_dir \
    --model_dump_dir path/to/your_ml_model_dump_dir




