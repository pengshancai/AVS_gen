import jsonlines
import csv
import argparse
import random
import json
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args


def extract_di(txt):
    start = txt.find("Discharge Instructions:")
    end = txt.find("Followup Instructions:")
    if start < 0 or end < 0:
        return ''
    di = txt[start: end].replace('\n',  ' ')
    di = ' '.join(di.split())
    return di


def extract_hc(txt):
    start = txt.find("Brief Hospital Course:")
    if start < 0:
        return ''
    end = txt.find("Medications on Admission:")
    if end == -1:
        end = txt.find("Discharge Medications:")
    if end == -1:
        end = txt.find("Discharge Disposition:")
    if end == 0:
        return ''
    assert start < end
    hc = txt[start: end].replace('\n',  ' ')
    hc = ' '.join(hc.split())
    return hc


def quality_check(txt):
    # Eliminate texts that are too short
    num_words = len(txt.split(' '))
    if num_words < 30:
        return False
    return True


def split_datasets(output_dir):
    with open(output_dir + 'all.json') as f:
        con = f.readlines()
    random.shuffle(con)
    num_train = math.floor(len(con) * 0.8)
    num_valid = math.floor(len(con) * 0.1)
    num_test = math.floor(len(con) * 0.1)
    with jsonlines.open(output_dir + 'train.json', 'w') as writer:
        for i in range(0, num_train):
            info = json.loads(con[i])
            writer.write(info)
    with jsonlines.open(output_dir + 'valid.json', 'w') as writer:
        for i in range(num_train, num_train + num_valid):
            info = json.loads(con[i])
            writer.write(info)
    with jsonlines.open(output_dir + 'test.json', 'w') as writer:
        for i in range(num_train + num_valid, num_train + num_valid + num_test):
            info = json.loads(con[i])
            writer.write(info)


if __name__ == "__main__":
    args = parse_args()
    with open(args.input_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        with jsonlines.open(args.output_path + 'all.json', mode='w') as writer:
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                cate, text = row[6], row[10]
                if cate != 'Discharge summary':
                    continue
                try:
                    hc = extract_hc(text)
                    di = extract_di(text)
                    if quality_check(di) and quality_check(hc):
                        writer.write({
                            "text": hc,
                            'summary': di
                        })
                except:
                    print('Errorous line:\t%s' % text)
    print('File processing finished, now randomly spliting into train/validation/test sets')
    split_datasets(args.output_dir)

