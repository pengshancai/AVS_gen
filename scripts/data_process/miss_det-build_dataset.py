import os
import json
from nltk.tokenize import TreebankWordTokenizer
from collections import defaultdict
from tqdm import tqdm
import argparse

selected_cates = {'sosy', 'dsyn', 'diap', 'phsu', 'orch', 'mobd',
                    'clnd', 'anab', 'bdsu', 'medd', 'patf', 'topp',
                    'inpo', 'fndg'}

ignored_cuis = {
    'C0332293', 'C0589120', 'C1457887', 'C0012634', 'C0277786', 'C3842676',
    'C0040808', 'C0489547', 'C0557061', 'C0184713', 'C0205250', 'C4534363',
    'C0234425', 'C0184511'
    }

ignored_findings = {"Abnormality", "Experimental Result", "Able (finding)", "Readiness", "good effect",
                    "Patient in hospital", "Present", "Wanted", "Agree", "One thing", "Used by",
                    "discharge diagnosis", "Probable diagnosis", "Postoperative diagnosis",
                    "treatment options", "Much better", "Agree", "Negative", "Positive", "Problem",
                    "Worse", "Willing", "Chief complaint (finding)", "Scheduled surgical procedure",
                    "Person 2", "Seen in department", "New finding since previous mammogram",
                    "Falls", "Frequent falls", "Finding", "Incidental Findings", "Negative Finding",
                    "Experimental Finding", "Physical findings", "Positive Finding", "disease history",
                    "Returned home", "A little bit", "Related personal status", "Alveolar volume"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_src_txt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input_src_json_dir",
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
    args = parser.parse_args()
    return args


def get_entity_span(ent):
    return ent['begin'], ent['end']


def get_token_tag(begin, end, entities_mapped):
    for (b, e), tag in entities_mapped.items():
        if int(b) <= begin and int(e) >= end:
            return tag
    return 0


def get_token_info(begin, end, ents):
    def get_single_token_info(ent, idx_ent):
        cates, cui = ent['sem_type'], ent['CUI']
        if cui in ignored_cuis:
            return 'none', -1, -1
        if '+' not in cates:
            cate = cates
            return cate, cui, idx_ent
        else:
            for cate in cates.split('+'):
                if cate in selected_cates:
                    return cate, cui, idx_ent
            return cates.split('+')[0], cui, idx_ent
    for idx_ent, ent in enumerate(ents):
        if int(ent['begin']) <= begin and int(ent['end']) >= end:
            return get_single_token_info(ent, idx_ent)
    return 'none', -1, -1


def get_token_ent(begin, end, entities_mapped, num_neg=None):
    for idx_ent, ((b_ent, e_ent), rtype) in enumerate(entities_mapped.items()):
        if int(b_ent) <= begin and int(e_ent) >= end:
            return idx_ent+1, num_neg
    num_neg += 1
    return 1000 + num_neg, num_neg


def rearrange_tags(tags):
    tags_ = []
    for i, tag in enumerate(tags):
        if i == 0:
            if tag == 0:
                tags_.append('O')
            elif tag == 1:
                tags_.append('B')
            elif tag == 2:
                tags_.append('B-SIM')
            elif tag == 3:
                tags_.append('B-IDT')
            else:
                assert False
        else:
            if tag == 0:
                tags_.append('O')
            elif tag == 1 and tags[i-1] == 1:
                tags_.append('I')
            elif tags[i] == 1 and tags[i-1] != 1:
                tags_.append('B')
            elif tags[i] == 2 and tags[i-1] == 2:
                tags_.append('I-SIM')
            elif tags[i] == 2 and tags[i-1] != 2:
                tags_.append('B-SIM')
            elif tags[i] == 3 and tags[i-1] == 3:
                tags_.append('I-IDT')
            elif tags[i] == 3 and tags[i-1] != 3:
                tags_.append('B-IDT')
            else:
                assert False
    return tags_


def rearrange_tokens(tokens, tags, cates, cuis):
    tokens_ = []
    tags_ = []
    cates_ = []
    assert len(tokens) == len(tags) == len(cates)
    for i, (token, tag, cate, cui) in enumerate(zip(tokens, tags, cates, cuis)):
        if tag > 0:
            tag = 1
        # add the token and tag
        tokens_.append(token)
        tags_.append(tag)
        if cate in selected_cates and cui not in ignored_cuis:
            cates_.append(cate)
        else:
            cates_.append('none')
    return tokens_, tags_, cates_


def get_ent_cates(ent):
    cate_str =  ent['sem_type']
    cates = cate_str.split('+')
    return cates


def judge_cate(cate, pref_term, required_cates):
    if cate in required_cates:
        return True
    elif cate == 'fndg':
        if pref_term in ignored_findings:
            return False
        else:
            return True
    else:
        return False


def get_required_entities(ents, required_cates = None):
    if not required_cates:
        # in phase 1, we use the following sem_types to select entities
        # required_cates1 = ['sosy', 'dsyn', 'diap', 'phsu', 'orch', 'clnd']
        # in phase 2, we add new sem_types
        required_cates = ['sosy', 'dsyn', 'diap', 'phsu', 'orch', 'mobd',
                          'clnd', 'anab', 'bdsu', 'medd', 'patf', 'topp',
                          'inpo']
    required_ents = []
    required_cuis = []
    for ent in ents:
        cates = get_ent_cates(ent)
        if 'color' in ent:
            required_ents.append(ent)
            required_cuis.append(ent['CUI'])
        for cate in cates:
            if judge_cate(cate, ent['pref_term'], required_cates):
                if ent['CUI'] not in required_cuis:
                    required_ents.append(ent)
                    required_cuis.append(ent['CUI'])
                break
    return required_ents


def get_relation_given_cui_pair(cui1, cui2):
    # TODO: this function is based on visiting database using sql queries
    # import mysql.connector
    # mydb = mysql.connector.connect(
    #     user='',
    #     password='',
    #     host='172.x.x.x',
    #     port=33xx,
    #     database='',
    #     auth_plugin='mysql_native_password'
    # )
    # mycursor = mydb.cursor()
    # mycursor.execute("SELECT * FROM MRREL WHERE cui1 = '%s' AND cui2 = '%s';" % (cui1, cui2))
    # recs = mycursor.fetchall()
    return None


def get_entity_pair_relation(ent1, ent2):
    '''
    :param ent1:
    :param ent2:
    :return:
    0 - the entities are not related
    1 - the entities are related, but no clear is_a relationship
    2 - the entities have is-a relationship
    3 - the two entities are identical
    '''
    cui1, cui2 = ent1['CUI'], ent2['CUI']
    if cui1 == cui2:
        return 3
    recs = get_relation_given_cui_pair(cui1, cui2)
    if len(recs) == 0:
        return 0
    for rec in recs:
        if rec[7] is None:
            continue
        if 'isa' in rec[7]:
            return 2
    return 1


def link_entities(ents_src, ents_tgt):
    '''
    The goal of this function is to discover entities from ents_src
    that are linked to entities from ents_tgt
    :return: A json file
    linked_entities1/2/3
    '''
    # entities of required types
    ents_src_r = get_required_entities(ents_src)
    ents_tgt_r = get_required_entities(ents_tgt)
    # entities of unrequired types
    ents_src_u = [ent for ent in ents_src if ent not in ents_src_r]
    ents_tgt_u = [ent for ent in ents_tgt if ent not in ents_tgt_r]
    cuis_src_r = [ent["CUI"] for ent in ents_src_r]
    cuis_tgt_r = [ent["CUI"] for ent in ents_tgt_r]
    ovlp = set(cuis_src_r).intersection(set(cuis_tgt_r))
    # for each entity in ents_src_r, find out if it is linked to an entity in ents_tgt_r
    linked_entities1 = []
    for i, ent_src in enumerate(ents_src_r):
        for j, ent_tgt in enumerate(ents_tgt_r):
            rel_type = get_entity_pair_relation(ent_src, ent_tgt)
            if rel_type > 0:
                linked_entities1.append((rel_type, ent_src, ent_tgt))
    # for each entity in ents_src_u, find out if it is linked to an entity in ents_tgt_r
    linked_entities2 = []
    for i, ent_src in enumerate(ents_src_u):
        for j, ent_tgt in enumerate(ents_tgt_r):
            rel_type = get_entity_pair_relation(ent_src, ent_tgt)
            if rel_type > 0:
                linked_entities2.append((rel_type, ent_src, ent_tgt))
    return linked_entities1 + linked_entities2


if __name__ == "__main__":
    args = parse_args()
    len_seq = defaultdict(list)
    for dtype in ['val', 'test', 'train']:
        print('Processing %s' % dtype)
        fnames = os.listdir(args.input_src_txt_dir)
        progress_bar = tqdm(range(len(fnames)))
        dataset = []
        for i in range(len(fnames)):
            try:
                fname = '%s.txt' % i
                data_out = []
                with open(args.input_src_txt_dir + fname) as f:
                    txt = f.read()
                with open(args.input_src_json_dir + fname + '.json') as f:
                    ents_src = json.load(f)
                with open(args.input_tgt_json_dir + fname + '.json') as f:
                    ents_tgt = json.load(f)
                entity_pairs = link_entities(ents_src, ents_tgt)
                entities_mapped = defaultdict(int)
                pairs = []
                for label, ent_src, ent_tgt in entity_pairs:
                    if label < 1:
                        continue
                    span = get_entity_span(ent_src)
                    entities_mapped[span] = max((entities_mapped[span], label))
                    idx_ent = ents_src.index(ent_src)
                    pairs.append((idx_ent, ent_tgt))
                # entities_mapped = {get_entity_span(ent_src): label for label, ent_src, _ in pairs_}
                if len(entities_mapped) == 0:
                    continue
                spans = list(TreebankWordTokenizer().span_tokenize(txt))
                tokens = []
                tags = []
                cates = []
                cuis = []
                idx_ents = []
                for begin, end in spans:
                    token = txt[begin: end]
                    cate, cui, idx_ent = get_token_info(begin, end, ents_src)
                    if cui != -1:
                        ner_tag = get_token_tag(begin, end, entities_mapped)
                    else:
                        ner_tag = 0
                    tokens.append(token)
                    tags.append(ner_tag)
                    cates.append(cate)
                    cuis.append(cui)
                    idx_ents.append(idx_ent)
                tokens, tags, cates = rearrange_tokens(tokens, tags, cates, cuis)
                tags_ = rearrange_tags(tags)
                info = {
                    'id': i,
                    'tokens': tokens,
                    'tags': tags_,
                }
                dataset.append(info)
            except:
                print('Failed id: %s' % i)
            _ = progress_bar.update(1)
        with open(args.output_dir + '%s.json' % dtype, 'w') as f:
            dataset_ = {
                "version": "missing event detection",
                "data": dataset
            }
            json.dump(dataset_, f)


