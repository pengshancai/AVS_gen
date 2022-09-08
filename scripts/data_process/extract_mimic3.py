import jsonlines
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_file",
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


if __name__ == "__main__":
    args = parse_args()
    with open(args.input_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        with jsonlines.open(args.output_file, mode='w') as writer:
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
    print('File Processing Finished.')



# with open('../AVS_gen_/data/sum/all.jsonl') as f:
#     con = f.readlines()
#
# import random
# import json
# random.shuffle(con)
# num_train = 25000
# num_valid = 3000
# num_test = 3082
# with jsonlines.open('../AVS_gen_/data/sum/train.jsonl', 'w') as writer:
#     for i in range(0, num_train):
#         info = json.loads(con[i])
#         writer.write(info)
#
# with jsonlines.open('../AVS_gen_/data/sum/valid.jsonl', 'w') as writer:
#     for i in range(num_train, num_train + num_valid):
#         info = json.loads(con[i])
#         writer.write(info)
#
# with jsonlines.open('../AVS_gen_/data/sum/test.jsonl', 'w') as writer:
#     for i in range(num_train + num_valid, num_train + num_valid + num_test):
#         info = json.loads(con[i])
#         writer.write(info)
