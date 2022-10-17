import argparse
from utils.simplification_utils import SARIsent
from readability import Readability
import rouge_score
from bert_score import score as bert_scorer
from tqdm import tqdm
import numpy as np


rouge_scorer = rouge_score.rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL'], use_stemmer=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--src_avs_path",
        type=str,
        default="",
        help="Path to the gold AVS file",
    )
    parser.add_argument(
        "--gen_avs_path",
        type=str,
        default="",
        help="Path to the generated AVS file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="pred",
        help="The evaluation target",
        choices=['src', 'gold', 'pred']
    )
    args = parser.parse_args()
    return args


def get_sari(src, pred, gold):
    return SARIsent(src, pred, [gold])


def get_dale_chall(pred):
    r = Readability(pred)
    return r.dale_chall()


def get_rouge_score(gold, pred):
    scores = rouge_scorer.score(gold, pred)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rouge3'].fmeasure, scores['rouge4'].fmeasure, \
           scores['rougeL'].fmeasure


def get_bert_score(gold, pred):
    P, R, F1 = bert_scorer([pred], [gold], lang='en', verbose=True)
    return P.item(), R.item(), F1.item()


def main():
    """
    Step 0: Preparation
    """
    args = parse_args()
    with open(args.src_avs_path) as f:
        golds = [rec['summmary'] for rec in con]
        srcs = [rec['text'] for rec in con]
    with open(args.gen_avs_path) as f:]
        preds = f.readlines()
    assert len(srcs) == len(golds) == len(preds)
    scores = defaultdict(list)
    progress = tqdm(range(len(srcs)))
    for src, gold, pred in zip(srcs, golds, preds):
        _ = progress.update(1)
        if args.target == 'pred':
            # Coverage
            rouge1, rouge2, rouge3, rouge4, rougeL = get_rouge_score(gold, pred)
            _, _, bf_f1 = get_bert_score(gold, pred)
            # Readability
            sari = get_sari(src, gold, pred)
            dale_chall = get_dale_chall(pred)
            # Other characteristics
            length = len(pred.split(' '))
            scores['rouge1'].append(rouge1)
            scores['rouge2'].append(rouge2)
            scores['rouge3'].append(rouge3)
            scores['rouge4'].append(rouge4)
            scores['rougeL'].append(rougeL)
            scores['bert_score'].append(bf_f1)
            scores['sari'].append(sari)
            scores['dale_chall'].append(dale_chall)
            scores['length'].append(length)
        else:
            if args.target == 'gold':
                target = gold
            else:
                target = src
            # Readability
            dale_chall = get_dale_chall(target)
            # Other characteristics
            length = len(target.split(' '))
            scores['dale_chall'].append(dale_chall)
            scores['length'].append(length)
    for metric, values in scores.items():
        print('%s:\t%s' % (metric, np.mean(values)))
    # This script may be further revised for analysis





