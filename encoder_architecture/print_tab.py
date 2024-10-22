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



results = json.load(open("outputs/results.json", "r"))


data = {}
tab = {}
pre_default = ["pre-context", "post-context"]
post_default = ['accuracy', "std", "BEFORE", "EQUAL", "AFTER", "IS_INCLUDED", "INCLUDES", "VAGUE"]
all_options = [x for x in pre_default]
for elm in results:
    all_options.extend(list(elm['config'].keys()))


all_options.extend(post_default)
to_discard = ['linear_layer', "lora", "lr1", 'lr2', 'context', "layers", 'init_weights', 'add_spacing', 'runs', 'epochs', 'patient', 'batch_size','cls_only','frozen', 'dual_optimizer', 'desc']

for x in all_options:
    if x not in to_discard and x not in tab:
        tab[x] = []
for elm in results:
    if "accuracy" in elm:
        for k in tab.keys():
            if k in to_discard:
                continue
        
            if k in elm['config']:
                if k == "layers":
                    if len(elm["config"]["layers"]) > 1:
                        tab[k].append("MLP")
                    else:
                        tab[k].append('Linear')
                elif k == 'context_size':
                    if elm['config']['context_size'] != None:
                        tab['pre-context'].append(elm['config']['context_size']['pre'])
                        tab['post-context'].append(elm['config']['context_size']['post'])
                else:
                    tab[k].append(elm["config"][k])
            elif k not in pre_default + post_default:
                tab[k].append(None)
                
        if 'context_size' not in elm['config']:
            tab['pre-context'].append(0)
            tab['post-context'].append(0)
        mean = round(np.asanyarray(elm["accuracy"]).mean(), 3) * 100
        std = round(np.asanyarray(elm["accuracy"]).std(), 3) * 100
        tab['accuracy'].append(mean)
        tab['std'].append(std)
        per_class = {}

        classes = ["BEFORE", "AFTER", "EQUAL", "IS_INCLUDED", "INCLUDES", "VAGUE"]
        for cls in classes:
            if cls in elm:
                if cls not in per_class:
                    per_class[cls] = []
                for rep in elm[cls]:
                    per_class[cls].append(rep["f1-score"])
            else:
                if cls not in per_class:
                    per_class[cls] = []
                per_class[cls].append(float("inf"))
                
        if 'SIMULTANEOUS' in elm:
            if 'EQUAL' not in per_class:
                per_class['EQUAL'] = []
            for rep in elm['SIMULTANEOUS']:
                per_class['EQUAL'].append(rep["f1-score"])
            

                        
        for k, v in per_class.items():
            tab[k].append(round(np.asanyarray(v).mean(), 3) * 100)
del tab['context_size']
for k, v in tab.items():
    print(k, len(v))
table = pd.DataFrame.from_dict(tab)
print(table.to_csv())


