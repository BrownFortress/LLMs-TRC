import argparse
import json
import numpy as np
import pickle
import os
from sandbox.data_structures import *
from utils.dataloader import Prompt, DataRow
from pprint import pprint
from collections import Counter
from sklearn.metrics import f1_score
import pandas as pd



parser = argparse.ArgumentParser(description="Train Model for Matres")
parser.add_argument("--results", type=str, help="", default="outputs")
parser.add_argument("--rels", type=str, help="", default="MATRES")
parser.add_argument("--skip_vague", help="", action="store_true")
args = parser.parse_args()

results = json.load(open(f"{args.results}/results.json", "r"))

rel_types = ["BEFORE", "AFTER", "EQUAL"]
if args.rels == "TB-DENSE":
    rel_types = ["BEFORE", "AFTER", "SIMULTANEOUS", "INCLUDES", "IS_INCLUDED"]
    if not args.skip_vague:
        print("Keep VAGUE")
        rel_types.append("VAGUE")
    else:
        print("Skip VAGUE")

print(rel_types)

data = {}
tab = {}
# pre_default = ["id", "pre-context", "post-context"]
pre_default = ["id"]
post_default = ["exec_runs", 'accuracy', "std"] + rel_types
all_options = [x for x in pre_default]
for elm in results:
    all_options.extend(list(elm['config'].keys()))

to_skip = ['context', 'desc', 'patience']
all_options.extend(post_default)
for x in all_options:
    if x not in to_skip and x not in tab:
        tab[x] = []

for elm in results:
    acc_vals = []
    if "micro avg" in elm:
        acc_vals.extend([scores["f1-score"] for scores in elm["micro avg"]])
    if "accuracy" in elm:
        acc_vals.extend(elm["accuracy"])
    if len(acc_vals) > 0:
        tab["id"].append(elm["id"])
        for k in tab.keys():
            if k in elm['config']:
                if k == "layers":
                    if len(elm["config"]["layers"]) > 1:
                        tab[k].append("MLP")
                    else:
                        tab[k].append('Linear')
                elif k == 'context_size':
                    if "context_size" in elm["config"] and elm['config']['context_size'] != None:
                        tab['pre-context'].append(elm['config']['context_size']['pre'])
                        tab['post-context'].append(elm['config']['context_size']['post'])
                else:
                    tab[k].append(elm["config"][k])
            elif k not in pre_default + post_default:
                tab[k].append(None)

        # if 'context_size' not in elm['config']:
        #     tab['pre-context'].append(0)
        #     tab['post-context'].append(0)
        tab["exec_runs"].append(len(acc_vals))
        mean = round(np.asanyarray(acc_vals).mean(), 3) * 100
        std = round(np.asanyarray(acc_vals).std(), 3) * 100
        tab['accuracy'].append(mean)
        tab['std'].append(std)
        per_class = {}
        # for k in elm:
        #     if k in rel_types:
        for k in rel_types:
            k_elm = k
            if k == "SIMULTANEOUS" and "EQUAL" in elm:
                k_elm = "EQUAL"
            if k not in per_class:
                per_class[k] = []
            if k_elm in elm:
                for rep in elm[k_elm]:
                    per_class[k].append(rep["f1-score"])
            else:
                per_class[k].extend([0 for _ in range(len(acc_vals))])

        for k, v in per_class.items():
            tab[k].append(round(np.asanyarray(v).mean(), 3) * 100)
# del tab['context_size']

table = pd.DataFrame.from_dict(tab)
print(table.to_csv())
