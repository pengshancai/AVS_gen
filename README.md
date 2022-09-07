# Generating After Visit Summaries

This is the repo for the COLING 2022 paper [Generation of Patient After-Visit Summaries to Support Physicians](Link will be available soon)

### Datasets

The model & results reported in this paper were obtained based on patient data from University of Massachusetts, Chan Medical School. Due to privacy issue, we could not open source our datasets. 

The good news is, we have recently found the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) contains data applicable to our task!

While MIMIC-III is a de-identified and publicly available dataset, you still need to apply to access the dataset on the [website](https://physionet.org/content/mimiciii/1.4/).

### Data Preprocess

#### Download & Preprocess MIMIC-III

Once you have been granted access to MIMIC-III, download [this file](https://physionet.org/content/mimiciii/1.4/NOTEEVENTS.csv.gz), which contains a large number of physician notes.

We will update the code for extracting discharge summary - AVS pairs from the downloaded file. 



### Training & Evaluating 

To train / evaluate the summarization model, run the following commnand:

```
python run_summarization.py \
    --model_name_or_path allenai/led-base-16384 \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file path/to/train.json \
    --validation_file .path/to/val.json \
    --test_file path/to/test.json \
    --output_dir path/to/output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --predict_with_generate \
    --num_train_epochs 6 \
    --save_steps 3000 \
    --save_total_limit 3 \
    --max_source_length 1024 \
    --max_target_length 256 \
```

To train / evaluate the hallucination detection model, run the following commnand:

```

```
