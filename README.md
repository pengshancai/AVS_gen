# Generating After Visit Summaries

This is the repo for the COLING 2022 paper [Generation of Patient After-Visit Summaries to Support Physicians](https://coling2022.org/) (Paper link will be available upon publication)

Note that the model & results reported in this paper were obtained based on patient data from University of Massachusetts, Chan Medical School. Due to privacy issue, we could not open source our datasets. 

The good news is, we have recently found the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) also contains data applicable to our task!

While MIMIC-III is a de-identified and publicly available dataset, you still need to apply to access the dataset on the [website](https://physionet.org/content/mimiciii/1.4/).

### Available Pre-trained Model

Our after-visit summary generation models pre-trained on MIMIC-III datasets is available [HERE](https://drive.google.com/file/d/1OvFUw0sqBJT-qOnokNkRwNr7LaN7YOHS/view?usp=sharing). To obtain the train/validation/test datasets, please refer to the instructions below.

[comment]: <> (### Datasets)
### Download & Preprocess MIMIC-III

Once you have been granted access to MIMIC-III, download [NOTEEVENTS.csv](https://physionet.org/content/mimiciii/1.4/NOTEEVENTS.csv.gz), which contains a table of de-identified physician notes.

Run the following command:
```
python scripts/data_process/extract_mimic3.py \
    --input_file path/to/NOTEEVENTS.csv \
    --output_dir path/to/output/data/directory
```

This command will extract hospital course - AVS pairs from the downloaded  file ```NOTEEVENTS.csv```, and randomly split them into train/validation/test datasets in the ```output_dir``` as you specified.


### Training & Evaluating the Summarization Model

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
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --predict_with_generate \
    --num_train_epochs 6 \
    --save_steps 3000 \
    --save_total_limit 3 \
    --max_source_length 1024 \
    --max_target_length 512 \
```

### Training & Evaluating the Error Detection Model

To train& apply our error detection model requires using [MetaMap](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap.html) to extract UMLS concepts from hospital course notes and AVS notes. 

#### Install MetaMap and Extract UMLS Concepts

Please install and set up MetaMap as [instructed](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/Installation.html), and then prepare the following folders:

- ```src_txt/``` contains hospital course notes, each hospital course note should be stored in a single txt file, e.g. ```src_txt_0.txt```;
- ```src_ent/``` contains UMLS concepts extracted by MetaMap from hospital course notes, the concepts extracted from a hospital course note txt file should be stored in a json file starting with the name of the txt file, e.g. ```src_txt_0.txt.json```;
- ```tgt_txt/``` contains AVS notes, similarly, each AVS note should be stored in a single txt file, e.g. ```tgt_txt_0.txt```;
- ```tgt_ent/``` contains UMLS concepts extracted by MetaMap from AVS notes, the naming criterion is similar to above , e.g. ```tgt_txt_0.txt.json```;

Specifically, the json file in ```src_ent/``` and ```tgt_ent/``` should adopt the following format:

```
[
    {"begin": "61", "end": "80", "CUI": "C0037278", "orgi_term": "of a skin infection", "pref_term": "Skin Diseases, Infectious", "sem_type": "dsyn"}, 
    {"begin": "94", "end": "105", "CUI": "C0003237", "orgi_term": "antibiotics", "pref_term": "Antibiotics, Antitubercular", "sem_type": "antb"},
    ...
]
```
where *begin* and *end* refers to the position where the UMLS concept begins and ends in the text, *CUI* is the unique concept identifier in the UMLS, *orgi_term* and *pref_term* are the original and preferred term of the concept, *sem_type* is the semantic type. All this information could be provided by MetaMap. 

To train / evaluate the hallucination detection model, you need to prepare the data through the following two steps:

1. To prepare the data for hallucination detection model, run the following command (You will need to revise a few hyper-parameters in the file before running it):
```
sh shell/prepare_data_err_detection.sh
```

2. To train the hallucination detection model, run the following commnand:
```
python ./run_err_det.py \
  --model_name_or_path google/bigbird-roberta-base \
  --train_file path/to/your_err_det_data_dir/train.json \
  --validation_file path/to/your_err_det_data_dir/val.json \
  --output_dir path/to/your_err_det_model_dump_dir \
  --save_steps 3000 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --num_train_epochs 3 \
  --hallucination_weight 3.0 
```
