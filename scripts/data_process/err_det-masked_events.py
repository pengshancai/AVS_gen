import random
import json
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
import re
import argparse
import jsonlines


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--input_txt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input_json_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_error_sents",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    return args


def select_ents(ents):
    ents_ = []
    ignored_cuis = {
        'C0332293', 'C0589120', 'C1457887', 'C0012634', 'C0277786', 'C3842676',
        'C0040808', 'C0489547', 'C0557061', 'C0184713', 'C0205250', 'C4534363',
        'C0701159', 'C3841449', 'C2825142', 'C1555319', 'C2926602', 'C1299581',
        'C0543467', 'C0087111', 'C0445223', 'C0150312', 'C0011900', 'C0011008',
        'C0809949',
    }
    # Ignore useless medical events
    for ent in ents:
        if ent['CUI'] in ignored_cuis:
            continue
        ents_.append(ent)
    return ents_


def txt2sents(txt):
    sents = []
    for start, end in PunktSentenceTokenizer().span_tokenize(txt):
        sent = txt[start: end]
        cur_start = 0
        se = re.search('\s\s', sent, cur_start)
        while se:
            cur_end = se.end() + cur_start
            sents.append([start + cur_start, start + cur_end, sent[cur_start: cur_end]])
            cur_start = cur_end
            se = re.search('\s\s', sent[cur_start:])
        sents.append([start + cur_start, end, sent[cur_start: end]])
    return sents


def get_sent_pairs(sents, ents):
    sent_pairs = []
    for k, (begin_sent, end_sent, sent) in enumerate(sents):
        if len(sent) < 30:
            continue
        if 'you were hospitalized because' in sent.lower():
            continue
        if 'your discharge diagnosis is' in sent.lower():
            continue
        if "significant Procedures and/or" in sent.lower():
            continue
        revisions = []
        for ent in ents:
            begin_ent, end_ent = int(ent['begin']), int(ent['end'])
            if begin_sent <= begin_ent and end_ent <= end_sent:
                # ent in sent
                revisions.append((begin_ent - begin_sent, end_ent - begin_sent))
        if len(revisions) == 0:
            continue
        revisions = random.choices(revisions, k=random.choice([1, 2]))
        revisions = sorted(revisions, key=lambda x: x[0])
        sent_ = ''
        cur_pos = 0
        for begin, end in revisions:
            sent_ += sent[cur_pos: begin]
            sent_ += '<mask>'
            cur_pos = end
        sent_ += sent[cur_pos: ]
        sent_pairs.append((sent, sent_))
    return sent_pairs


def split_datasets(fnames, num_folds):
    fnames.sort()
    total_len = len(fnames)
    fold_len = int(total_len / 5)
    fnames_by_folds = []
    for i in range(num_folds):
        fnames_by_folds.append([fnames[j] for j in range(i * fold_len, (i + 1) * fold_len)])
    return fnames_by_folds


def write_instance(fname, args, writer):
    with open(args.input_txt_dir + fname) as f:
        txt = f.read().strip()
    with open(args.input_json_dir + fname + '.json') as f:
        ents = select_ents(json.load(f))
    sents = txt2sents(txt)
    sent_pairs = get_sent_pairs(sents, ents)
    for sent, sent_ in sent_pairs:
        writer.write({
            'text': sent_,
            'summary': sent
        })


if __name__ == "__main__":
    args = parse_args()
    fnames = os.listdir(args.input_txt_dir)
    fnames_by_folds = split_datasets(fnames, args.num_folds)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for fold in range(args.num_folds):
        print('Processing Fold %s' % fold)
        fnames_inference = fnames_by_folds[fold]
        fnames_train = []
        for fold_ in range(args.num_folds):
            if fold != fold_:
                fnames_train += fnames_inference[fold_]
        output_dir_fold = args.output_dir + 'fold_%s/' % fold
        if not os.path.exists(output_dir_fold):
            os.mkdir(output_dir_fold)
        # output the train file for this fold
        with jsonlines.open(output_dir_fold + 'train.json') as writer:
            for fname in fnames_train:
                write_instance(fname, args, writer)
        # output the inference file for this fold
        with jsonlines.open(output_dir_fold + 'inference.json') as writer:
            for fname in fnames_train:
                write_instance(fname, args, writer)

