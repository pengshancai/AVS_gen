import random
from nltk.tokenize import WhitespaceTokenizer
import json
from nltk.tokenize.punkt import PunktSentenceTokenizer
import argparse
import numpy as np
import re
import os
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
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
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--input_src_txt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input_tgt_txt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input_tgt_json_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_dump_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args


def get_available_sent_ids(sents, ents):
    ids = []
    # we require the available sentence to fulfill the following conditions:
    # 1. Length requirement
    # 2. Does not contain the some phrases
    # 3. At least contain one entity
    for idx, (begin_sent, end_sent, sent) in enumerate(sents):
        revisions = []
        for ent in ents:
            begin_ent, end_ent = int(ent['begin']), int(ent['end'])
            if begin_sent <= begin_ent and end_ent <= end_sent:
                # ent in sent
                revisions.append((begin_ent - begin_sent, end_ent - begin_sent))
                break
        if len(revisions) != 0:
            ids.append(idx)
    return ids


def get_masked_sent(begin_sent, end_sent, sent, ents):
    revisions = []
    for ent in ents:
        begin_ent, end_ent = int(ent['begin']), int(ent['end'])
        if begin_sent <= begin_ent and end_ent <= end_sent:
            # ent in sent
            revisions.append((begin_ent - begin_sent, end_ent - begin_sent))
    if len(revisions) == 0:
        return sent, 0
    revisions = random.choices(revisions, k=random.choice([1,2]))
    revisions = sorted(revisions, key=lambda x: x[0])
    sent_ = ''
    cur_pos = 0
    for begin, end in revisions:
        sent_ += sent[cur_pos: begin]
        sent_ += '<mask>'
        cur_pos = end
    sent_ += sent[cur_pos:]
    return sent_, len(revisions)


def get_corrupted_tokens_and_labels(sent1, sent2, ws_tokenizer):
    tokens1 = ws_tokenizer.tokenize(sent1)
    tokens2 = ws_tokenizer.tokenize(sent2)
    m, n = len(tokens1), len(tokens2)
    L = np.array([[0 for x in range(n + 1)] for y in range(m + 1)])
    # Following steps build L[m+1][n+1] in bottom up fashion. Note that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif tokens1[i - 1] == tokens2[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    # Following code is used to print LCS
    index = L[m][n]
    # Create a character array to store the lcs string
    lcs = [""] * (index+1)
    lcs[index] = ""
    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    labels = []
    while i > 0 and j > 0:
        # If current character in X[] and Y are same, then current character is part of LCS
        if tokens1[i - 1] == tokens2[j - 1]:
            lcs[index - 1] = tokens1[i - 1]
            i -= 1
            j -= 1
            index -= 1
            labels.insert(0, 'O')
        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
            labels.insert(0, 'R1')
    assert len(tokens2) == len(labels)
    return tokens2, labels


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


def split_datasets(fnames, num_folds):
    fnames.sort()
    total_len = len(fnames)
    fold_len = int(total_len / 5)
    fnames_by_folds = []
    for i in range(num_folds):
        fnames_by_folds.append([fnames[j] for j in range(i * fold_len, (i + 1) * fold_len)])
    return fnames_by_folds


def select_ents(ents):
    ents_ = []
    ignored_cuis = {
        'C0332293', 'C0589120', 'C1457887', 'C0012634', 'C0277786', 'C3842676',
        'C0040808', 'C0489547', 'C0557061', 'C0184713', 'C0205250', 'C4534363',
        'C0701159', 'C3841449', 'C2825142', 'C1555319', 'C2926602', 'C1299581',
        'C0543467', 'C0087111', 'C0445223', 'C0150312', 'C0011900', 'C0011008',
        'C0809949',
    }
    for ent in ents:
        if ent['CUI'] in ignored_cuis:
            continue
        ents_.append(ent)
    return ents_


def get_correct_tokens_and_labels(sent, ws_tokenizer):
    tokens = ws_tokenizer.tokenize(sent)
    labels = ['O' for _ in tokens]
    return tokens, labels


def main(args):
    device = torch.device('cuda:%s' % args.device_id)
    ws_tokenizer = WhitespaceTokenizer()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    fnames = os.listdir(args.input_tgt_txt_dir)
    fnames_by_folds = split_datasets(fnames, args.num_folds)
    dataset = []
    for fold in range(5):
        print('Processing Fold %s' % fold)
        model = BartForConditionalGeneration.from_pretrained(args.model_dump_dir + 'fold_%s/' % fold).to(device)
        tokenizer = BartTokenizer.from_pretrained(args.model_dump_dir + 'fold_%s/' % fold)
        fnames_inference = fnames_by_folds[fold]
        progress_bar = tqdm(range(len(fnames_inference)))
        for fname in fnames_inference:
            with open(args.input_src_txt_dir + fname) as f:
                txt_src = f.read().strip()
            with open(args.input_tgt_txt_dir + fname) as f:
                txt_tgt = f.read().strip()
            with open(args.input_tgt_json_dir + fname + '.json') as f:
                ents = select_ents(json.load(f))
            sents = txt2sents(txt_tgt)
            ids_sent_avlb = get_available_sent_ids(sents, ents)
            ids_sent_revise = random.sample(ids_sent_avlb, k=min(len(ids_sent_avlb), args.num_error_sents))
            idx2sent_currputed = {}
            for idx_sent in ids_sent_revise:
                begin_sent, end_sent, sent = sents[idx_sent]
                resemble_score, num_mask = 1.0, 0
                for attempt in range(3):
                    masked_sent, num_mask = get_masked_sent(begin_sent, end_sent, sent, ents)
                    batch = tokenizer(masked_sent, return_tensors='pt').to(device)
                    generated_ids = model.generate(batch['input_ids'], max_length=256).cpu().numpy()[0]
                    generated_sent = tokenizer.decode(generated_ids).replace('</s>', '').replace('<s>', '')
                    resemble_score = scorer.score(sent, generated_sent)['rougeL'].fmeasure
                    if resemble_score < 0.99:
                        break
                # we require the generated sents do not resemble the original sents
                if resemble_score >= 0.99 or num_mask == 0:
                    continue
                try:
                    tokens_generated, labels = get_corrupted_tokens_and_labels(sent, generated_sent)
                    idx2sent_currputed[idx_sent] = (tokens_generated, labels)
                except:
                    continue
            tokens_tgt = []
            labels_tgt = []
            for idx_sent, (_, _, sent) in enumerate(sents):
                if idx_sent in idx2sent_currputed:
                    tokens_sent, labels_sent = idx2sent_currputed[idx_sent]
                else:
                    tokens_sent, labels_sent = get_correct_tokens_and_labels(sent, ws_tokenizer)
                tokens_tgt += tokens_sent
                labels_tgt += labels_sent
            tokens_src, labels_src = get_correct_tokens_and_labels(txt_src, ws_tokenizer)
            info = {
                'id': fname,
                'tokens': tokens_tgt + [' [SEP] '] + tokens_src,
                'tags': labels_tgt + ['O'] + labels_src,
            }
            dataset.append(info)
            _ = progress_bar.update(1)
    train_set_size = int(len(dataset) * 0.9)
    with open(args.output_dir + 'train.json', 'w') as f:
        dataset_ = {
            "version": "train",
            "data": dataset[: train_set_size]
        }
        json.dump(dataset_, f)
    with open(args.output_dir + 'valid.json', 'w') as f:
        dataset_ = {
            "version": "valid",
            "data": dataset[train_set_size: ]
        }
        json.dump(dataset_, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
