import os, sys, json, pickle
import os.path as osp
import random as rd
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import time
import ipdb
import string

fout = None

# Utils
def double_log(msg, fout=None):
    print(msg)
    if fout is not None:
        fout.write(str(msg) + '\n')
        fout.flush()

def dump(data, f):
    def dump_pkl(data, pth):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth):
        json.dump(data, open(pth, 'w'))

    def dump_jsonl(data, f):
        lines = [json.dumps(x, ensure_ascii=False) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f):
        data.to_excel(f, index=False)

    def dump_csv(data, f):
        data.to_csv(f, index=False)

    def dump_tsv(data, f):
        data.to_csv(f, sep='\t', index=False)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f)

def load(f):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f) 

# Accuracy Report
def report_acc(df, group='category'):
    assert 'split' in df
    assert group in [None, 'category', 'l2-category']

    res = defaultdict(list)
    res['split'] = ['full', 'dev', 'test']
    if group is None:
        res['overall'] = [np.mean(df['hit']), np.mean(df[df['split'] == 'dev']['hit']), np.mean(df[df['split'] == 'test']['hit'])]
        return pd.DataFrame(res)
    
    elif group in df:
        abilities = list(set(df[group]))
        abilities.sort()
        for ab in abilities:
            sub_df = df[df[group] == ab]
            res[ab] = [np.mean(sub_df['hit']), np.mean(sub_df[sub_df['split'] == 'dev']['hit']), np.mean(sub_df[sub_df['split'] == 'test']['hit'])]
        return pd.DataFrame(res)

def build_choices(item):
    ret = {}
    for ch in 'ABCD':
        if not pd.isna(item[ch]):
            ret[ch] = item[ch]
    return ret

# Prefetch Answers
def can_infer_option(answer, num_choice=5):
    choices = string.ascii_uppercase[:num_choice]
    if 'Failed to obtain answer via API' in answer:
        return False
    
    def count(splits, choices='ABCD', prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    splits = [x.strip() for x in answer.split()]
    if count(splits, choices) == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                double_log(f'A might be a quantifier in the string: {answer}. ', fout)
                break
            if ch in splits:
                return ch 
    tups = [
        ('', '.'), ('', ','), ('', ':'), ('', ')'), ('', ').'), ('(', ')'), 
        ('(', ').'), (':', ''), (':', ','), (':', '.'), (':', ')'), (':', ').')
    ]
    for tup in tups:
        if count(splits, choices, prefix=tup[0], suffix=tup[1]) == 1:
            for ch in choices:
                if tup[0] + ch + tup[1] in splits:
                    return ch
    return False

def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in 'ABCD'
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer(answer, choices):
    copt = can_infer_option(answer)
    return copt if copt else can_infer_text(answer, choices)

def prefetch_answer(item):
    choices = build_choices(item)
    return can_infer(item['prediction'], choices)

            
# Extract answer from multiple rolling records
def eval_sub_data(sub_data, answer_map):
    lt = len(sub_data)
    GT, PRED = [], []
    for i in range(lt):
        item = sub_data.iloc[i]
        idx = item['index']
        GT.append(answer_map[idx])
        PRED.append(prefetch_answer(item))
        if PRED[-1] and (GT[-1] != PRED[-1]):
            return 0
    
    for i in range(lt):
        if PRED[i]:
            continue
        else:
            # ret, _ = extract_answer_from_item(model, sub_data.iloc[i])
            ret = sub_data.iloc[i]['prediction']
            PRED[i] = ret
            # if PRED[i] != GT[i]:
            if GT[i] not in PRED[i]:
                return 0
    return 1
    
# Evaluate Results
def eval_result(pred_file, gt_file, tmp_dir):
    rd.seed(2680)
    double_log(f'Evaluating {pred_file}', fout)

    result_file = pred_file.replace('.xlsx', f'_result.pkl')
    result = {}
    if osp.exists(result_file):
        result = load(result_file)
        
    data = load(pred_file)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]
    for k in data.keys():
        data[k.lower() if k not in 'ABCD' else k] = data.pop(k)

    meta = load(gt_file)

    data_main = data[data['index'] < int(1e6)]
    cate_map = {i: c for i, c in zip(meta['index'], meta['category'])}
    l2_cate_map = {i: c for i, c in zip(meta['index'], meta['l2-category'])}
    split_map = {i: c for i, c in zip(meta['index'], meta['split'])}
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}
    
    lt = len(data_main)
    hit, tot = 0, 0

    for i in tqdm(range(lt)):
        # Dealing with the normal part
        item_main = data_main.iloc[i]
        idx = item_main['index']

        if idx in result:
            correct = result[idx]
            assert correct in [0, 1]
            hit += correct
            tot += 1
            continue
        
        sub_data = data[data['index'] % int(1e6) == idx]
        ret = eval_sub_data(sub_data, answer_map)
        result[idx] = ret
        hit += ret
        tot += 1
        
        dump(result, result_file)

        if (i + 1) % 10 == 0:
            double_log(f'Evaluating {pred_file}: {i + 1}/{lt}, Acc: {hit / tot * 100: .2f}%. ', fout)

    dump(data_main, f'{tmp_dir}/tmp.xlsx')
    data_main = load(f'{tmp_dir}/tmp.xlsx')   
    
    res = load(result_file)
    indices = data_main['index']
    data_main['hit'] = [res[i] for i in indices]
    data_main['split'] = [split_map[i] for i in indices]
    main_idx = data_main['index']
    data_main['category'] = [cate_map[i] for i in main_idx]
    data_main['l2-category'] = [l2_cate_map[i] for i in main_idx]
    
    # load split
    dump(data_main, pred_file.replace('.xlsx', f'_result.xlsx'))
    data_main = load(pred_file.replace('.xlsx', f'_result.xlsx'))
    
    overall = report_acc(data_main, None)
    dump(overall, pred_file.replace('.xlsx', f'_overall.csv'))
    double_log(overall)

    l2 = report_acc(data_main, 'l2-category')
    dump(l2, pred_file.replace('.xlsx', f'_l2.csv'))
    double_log(l2)

    leaf = report_acc(data_main, 'category')
    dump(leaf, pred_file.replace('.xlsx', f'_leaf.csv'))
    double_log(leaf)
    
    if fout is not None:
        fout.close()

    return overall, l2, leaf

if __name__ == '__main__':
    acc, l2, leaf = eval_result(
        pred_file='/data/hypertext/lucaszhao/MMGPT-PyTorch/checkpoints/pretrain-clip-large+conv1+baichuan2-7b-pair20m-v1-sft-llava665k/mmbench.xlsx', 
        gt_file='/data/hypertext/yancie/dataset/MMBench/mmbench_dev_cn_20231003.tsv', 
        tmp_dir='/data/hypertext/lucaszhao/MMGPT-PyTorch/checkpoints/pretrain-clip-large+conv1+baichuan2-7b-pair20m-v1-sft-llava665k/'
    )
    