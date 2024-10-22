import pickle
import os
from pprint import pprint
from sandbox.data_structures import *
from collections import Counter

data_base_path = "data/TB-DENSE"
splits = {}

def perc_dict(dictionary, decimal=3):
    return {k: round(v/ sum(dictionary.values()), decimal) * 100 for k, v in dictionary.items()}


for file in os.listdir(data_base_path):
    root_name = file.split(".")[0]
    if root_name in ["train", "valid", "test"]:
        print(os.path.join(data_base_path, file))
        with open(os.path.join(data_base_path, file), "rb") as reader:
            splits[root_name] = pickle.load(reader)
            

diff_sents = {}
diff_paragraphs = {}
num_relations  = {}
docs = {}
for partition, documents in splits.items():
    num_relations[partition] = []
    docs[partition] = []
    for doc in documents:
        d_id = doc.d_id
        docs[partition].append(d_id)
        for rel_type, rels  in doc.temporal_relations.items():
            # if rel_type != "VAGUE":
            for rel in rels:
                num_relations[partition].append(rel_type)
                event1 = rel.event1
                event2 = rel.event2
                diff_sent = abs(event1.sent_id - event2.sent_id)
                # diff_para = abs(event1.paragraph_id - event2.paragraph_id)
                if diff_sent not in diff_sents:
                    diff_sents[diff_sent] = 0
                    
                # if diff_para not in diff_paragraphs:
                    # diff_paragraphs[diff_para] = 0
                    
                diff_sents[diff_sent] += 1
                # diff_paragraphs[diff_para] += 1
    print(partition)
    pprint(perc_dict(diff_sents))
    # pprint(perc_dict(diff_paragraphs))
    
for p, v in num_relations.items():
    print(p)
    print(Counter(v), "SUM:", len(v))
    
for p, v in docs.items():
    print(p)
    print("DOCs", len(set(v)))
print(len(set(docs['train']).union(docs['valid'])))
assert len(set(docs['test']).intersection(set(docs['train']).union(docs['valid']))) == 0    