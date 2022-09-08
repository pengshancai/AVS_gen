import random
from nltk.tokenize import WhitespaceTokenizer
from collections import defaultdict
import json
from tqdm import tqdm
from nltk.tokenize.punkt import PunktSentenceTokenizer
import os
import numpy as np
import re
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
from utils.metamap_utils import extract_entities_metamap


'''
For each AVS note, make three replacing mistakes
Specifically, compared to 8.1.5, this version annotates missing error as such
    1. Find all the sent_src related to sent_tgt
'''

path_in_txt = './data/src-tgt_sep/%s-target_txt/'
path_in_ents = './data/src-tgt_sep/%s-target_json/'
path_out = './data/json/src-tgt_9.5.1/'

if not os.path.exists(path_out):
    os.mkdir(path_out)


MAX_MASK_PER_SENT = 2
NUM_FOLDS = 5

dtype = 'train'
total_len = len([fname for fname in os.listdir(path_in_txt % dtype)])
fold_len = int(total_len / 5)
ids_by_folds = [list(range(i*fold_len, (i+1)*fold_len)) for i in range(NUM_FOLDS)]
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
ws_tokenizer = WhitespaceTokenizer()


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


'''
Train
'''
for fold in range(NUM_FOLDS):
    print('Processing Fold %s' % fold)
    ids_dev = ids_by_folds[fold]
    ids_train = []
    for fold_ in range(NUM_FOLDS):
        if fold != fold_:
            ids_train += ids_by_folds[fold_]
    path_out_i = path_out + 'fold_%s/' % fold
    if not os.path.exists(path_out_i):
        os.mkdir(path_out_i)
    instances_train = []
    for idx in ids_train:
        with open(path_in_txt % dtype + '%s.txt' % idx) as f:
            txt = f.read().strip()
        with open(path_in_ents % dtype + '%s.txt.json' % idx) as f:
            ents = select_ents(json.load(f))
        sents = txt2sents(txt)
        sent_pairs = get_sent_pairs(sents, ents)
        for sent, sent_ in sent_pairs:
            instances_train.append(
                    {
                        'id': idx,
                        'text': sent_,
                        'summary': sent
                    }
                )
    instances_dev = []
    for idx in ids_dev:
        with open(path_in_txt % dtype + '%s.txt' % idx) as f:
            txt = f.read().strip()
        with open(path_in_ents % dtype + '%s.txt.json' % idx) as f:
            ents = select_ents(json.load(f))
        if len(txt) < 30:
            continue
        sents = txt2sents(txt)
        sent_pairs = get_sent_pairs(sents, ents)
        for sent, sent_ in sent_pairs:
            instances_dev.append(
                    {
                        'id': idx,
                        'text': sent_,
                        'summary': sent
                    }
                )
    with open(path_out_i + 'train.json', 'w') as f:
        dataset_train = {
            'data': instances_train,
            'version': '9.5.1'
        }
        json.dump(dataset_train, f)
    with open(path_out_i + 'dev.json', 'w') as f:
        dataset_dev = {
            'data': instances_dev,
            'version': '9.5.1'
        }
        json.dump(dataset_dev, f)


'''
Generate
'''
idx_device = 0
fold = 0

NUM_ERROR_SENTS = 5

device = torch.device('cuda:%s' % idx_device)
model = BartForConditionalGeneration.from_pretrained("./dump/mlm_9.5.1_fold_%s" % fold).to(device)
tokenizer = BartTokenizer.from_pretrained("./dump/mlm_9.5.1_fold_%s" % fold)


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


def get_corrupted_tokens_and_labels(sent1, sent2):
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


def get_correct_tokens_and_labels(sent):
    tokens = ws_tokenizer.tokenize(sent)
    labels = ['O' for _ in tokens]
    return labels


# for fold in range(5):
print('Processing Fold %s' % fold)
ids_dev = ids_by_folds[fold]
instances_dev = []
for idx_doc in ids_dev:
    with open(path_in_txt % dtype + '%s.txt' % idx_doc) as f:
        txt = f.read().strip()
    # with open(path_in_ents % dtype + '%s.txt.json' % idx_doc) as f:
    #     ents = select_ents(json.load(f))
    ents = select_ents(annotate_metamap(txt))
    sents = txt2sents(txt)
    ids_sent_avlb = get_available_sent_ids(sents, ents)
    ids_sent_revise = random.sample(ids_sent_avlb, k=min(len(ids_sent_avlb), NUM_ERROR_SENTS))
    idx2sent_currputed = {}
    for idx_sent in ids_sent_revise:
        begin_sent, end_sent, sent = sents[idx_sent]
        resemble_score, num_mask = 1.0, 0
        for attempt in range(3):
            masked_sent, num_mask = get_masked_sent(begin_sent, end_sent, sent, ents)
            batch = tokenizer(masked_sent, return_tensors='pt').to(device)
            generated_ids = model.generate(batch['input_ids'], max_length=256).cpu().numpy()[0]
            generated_sent = tokenizer.decode(generated_ids).replace('</s>', '').replace('<s>', '')
            # print(sent)
            # print(masked_sent)
            # print(generated_sent)
            resemble_score = scorer.score(sent, generated_sent)['rougeL'].fmeasure
            if resemble_score < 0.99:
                break
        # we require the generated sents do not resemble the original sents
        if resemble_score >= 0.99 or num_mask == 0:
            continue
        tokens_generated, labels = get_corrupted_tokens_and_labels(sent, generated_sent)
        idx2sent_currputed[idx_sent] = (tokens_generated, labels)
    tokens_note = []
    labels_note = []
    for idx_sent, (_, _, sent) in enumerate(sents):
        if idx_sent in idx2sent_currputed:
            (tokens_sent, labels_sent) = idx2sent_currputed[idx_sent]
            tokens_note += tokens_sent
            labels_note += labels_sent





