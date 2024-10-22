
import json
import pickle
from pprint import pprint
import os

from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    BertTokenizer,
    BertModel,
)

import copy


from captum.attr import LayerIntegratedGradients, KernelShap
from functools import partial
from collections import Counter
import numpy as np

from dataclasses import dataclass
import math
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F, torch.nn as nn
from tqdm import tqdm

from sandbox.data_structures import Event, Document, Relation, TemporalFunction
from utils.dataloader_bis import TempDataset, collate_fn, Prompter
from utils.model_bertish import BERT_Temp_Analysis, FeedForward
from utils.attribution_scores import add_token, aggregate, construct_references
import argparse


parser = argparse.ArgumentParser(description="Train Model for Matres")
parser.add_argument("--config_file", type=str, help="add_path_to_the_config_file", default="configs/word_conf_linear_dual.json")
parser.add_argument("--device", type=str, help="add_cuda", default="cuda:0")
parser.add_argument("--dataset", type=str, help="add a dataset", default="MATRES", required=True, choices=[ "TIMELINE", "MATRES", "TB-DENSE"])
parser.add_argument("--n_steps", type=int, help="add a steps KShap", default=256)

args = parser.parse_args()

DEVICE = args.device
N_STEPS = args.n_steps

config = json.load(open(args.config_file, "r"))


# DATABASE_PATH = "../beast/TempRel/data/MATRES_NEW_NLTK"
DATASET_NAME = args.dataset
DATABASE_PATH = os.path.join("data", DATASET_NAME)

BASE_MODEL_PATH = "bin"
splits = {}

for file in os.listdir(DATABASE_PATH):
    root_name = file.split(".")[0]
    if root_name in ["train", "valid", "test"]:
        print(os.path.join(DATABASE_PATH, file))
        with open(os.path.join(DATABASE_PATH, file), "rb") as reader:
            splits[root_name] = pickle.load(reader)

tokenizer = RobertaTokenizerFast.from_pretrained(config["model_name"])
raw_model = RobertaModel.from_pretrained(config["model_name"])

datasets = {}
prompt_maker = Prompter(tokenizer)
if config["cls_only"]:
    if config["context"] == "sentence":
        prompt = prompt_maker.events_eos
    elif config["context"] == "paragraph":
        prompt = prompt_maker.events_eos_paragraph
    else:
        print("Prompt not found!")

else:
    prompt = prompt_maker.word_in_sentence

datasets["train"] = TempDataset(splits["train"], prompt_function=prompt)

label_mapping = datasets["train"].dict_rel


datasets["valid"] = TempDataset(
    splits["valid"], prompt_function=prompt, lab_dict=label_mapping
)
datasets["test"] = TempDataset(
    splits["test"], prompt_function=prompt, lab_dict=label_mapping
)


train_loader = DataLoader(
    datasets["train"],
    batch_size=config["batch_size"],
    collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE),
    shuffle=True,
)

dev_loader = DataLoader(
    datasets["valid"],
    batch_size=64,
    collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE),
)

test_loader = DataLoader(
    datasets["test"],
    batch_size=1,
    collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE),
)

classfier = FeedForward(config["layers"], len(label_mapping))
model = BERT_Temp_Analysis(raw_model, classifier=classfier, text_cls=config["cls_only"])
if DATASET_NAME == "MATRES":
    MODEL_ID = "2024-05-29|16:27:28.223719_XNZDZQW8CZ4GS0202BPSIUH8CVUH0MAT"
elif DATASET_NAME == "TIMELINE":
    MODEL_ID = "2024-06-05|17:20:19.982216_O7A3XEZNR7XZV16G3VGLSKFRBY5K45NS" 
elif DATASET_NAME == "TB-DENSE":
    MODEL_ID = "2024-06-05|17:14:25.181490_RNAWSNVY5P4KPANHX6FZ62Z71S8FH7GP"
    

model.load_state_dict(torch.load("bin/" + MODEL_ID))

# add_token(tokenizer, '[ABLATE_WORD]', model)

ks = KernelShap(model)

global_attributes = []
wrong_predictions = []
corpus_tokens = []
model.to(DEVICE)

@dataclass
class BertAttribute:
    seq_attr: torch.Tensor
    input_tokens: torch.Tensor
    
    
for sample in tqdm(test_loader):
    model.zero_grad()
    # baseline = torch.LongTensor(construct_references(sample['input_ids'], tokenizer)).to(DEVICE)
    with torch.no_grad():
        logits = model(sample['input_ids'], sample['attention_mask'], sample['masks'], sample['complementary_mask'], sample['to_duplicate'])
        tmp_pred_rel = np.argmax(logits.cpu().numpy(), axis=1)
        gt = sample['labels'].cpu().numpy()
    assert len(gt) == 1
    model.zero_grad()
    attr_tmp = ks.attribute(
        sample['input_ids'],
        target=sample['labels'].to(DEVICE),
        n_samples=300,
        additional_forward_args=(sample['attention_mask'], sample['masks'], sample['complementary_mask'], sample['to_duplicate'])
        ) 
    attr = BertAttribute(attr_tmp[0].cpu(), sample['input_ids'][0].cpu())
 
    
    if gt[0] == tmp_pred_rel[0]:
        global_attributes.append(attr)
        wrong_predictions.append(None)
    else:
        wrong_predictions.append(attr)
        global_attributes.append(None)
        


res = {"good":global_attributes, "bad":wrong_predictions, "wierd":None}            
with open(DATASET_NAME+"_kernelShap_RoBERTa.pkl", "wb") as file:
    file.write(pickle.dumps(res))