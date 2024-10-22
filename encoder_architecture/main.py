import argparse
from datetime import datetime
import pickle
import os
import json
import math
from pprint import pprint
import random
import string

from functools import partial
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from filelock import Timeout, FileLock
import sys 
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    BertTokenizer,
    BertModel,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    get_linear_schedule_with_warmup,
)

from utils.support import train_loop, eval_loop, init_weights
from utils.dataloader import (
    TempDataset,
    collate_fn,
    Prompter,
    TempDatasetFrozen,
    collate_fn_frozen,
)
from utils.model_bertish import (
    BERT_Temp,
    FeedForward,
    LoRaModel,
    EncoderFrozen,
    WrapperClassifier,
)
from sandbox.data_structures import Event, Document, Relation, TemporalFunction

from random import randint
from time import sleep

sleep(randint(1,30))

parser = argparse.ArgumentParser(description="Train Model for Matres")

parser.add_argument("--config_file", type=str, help="add_path_to_the_config_file")
parser.add_argument(
    "--save_model", action="store_true", help="Do you want to save the models?"
)

parser.add_argument(
    "--not_skip_vague", action="store_true", help="Do you want to skip vague?"
)

parser.add_argument("--device", type=str, help="add_cuda")
parser.add_argument("--model_name", type=str, help="add_cuda")
parser.add_argument("--dataset", type=str, help="add a dataset", required=True, choices=["MATRES", "TIMELINE", "MATRES_NEW", "MATRES_ERE", "TB-DENSE"])


args = parser.parse_args()
config = json.load(open(args.config_file, "r"))

FROZEN_DUMP_PATH = 'frozen_embeddings/'
RESULT_PATH = "outputs/results.json"
RESULT_PATH_LOCK = "outputs/results.lock"

# ================= Paths creations =======================
if not os.path.exists("bin/"):
    os.mkdir("bin/")
if not os.path.exists("outputs/"):
    os.mkdir("outputs/")
if not os.path.exists(FROZEN_DUMP_PATH):
    os.mkdir(FROZEN_DUMP_PATH)
# ================= Init Save results ====================

lock = FileLock(RESULT_PATH_LOCK, timeout=50)

if not os.path.isfile(RESULT_PATH):
    with lock:
        with open(RESULT_PATH, "w") as f:
            f.write(json.dumps([], indent=4))
with open(RESULT_PATH, "r") as f:
    results = json.loads(f.read())
id_res = (
    datetime.now().isoformat("|")
    + "_"
    + "".join(random.choices(string.ascii_uppercase + string.digits, k=32))
)

if args.model_name is not None:
    config['model_name'] = args.model_name
    
config['dataset_name'] = args.dataset
config['not_skip_vague'] = args.not_skip_vague
new_row = {"id": id_res, "created_at": datetime.now().isoformat("|"), "config": config}
results.append(new_row)
with lock:
    with open(RESULT_PATH, "w") as f:
        f.write(json.dumps(results, indent=4))

# ================= Init Model ============================
if "roberta" in config["model_name"]:
    print("Im using ROBERTA")
    tokenizer = RobertaTokenizerFast.from_pretrained(config["model_name"])
    raw_model = RobertaModel.from_pretrained(config["model_name"])
elif "bert" in config["model_name"]:
    print("Im using BERT")
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    raw_model = BertModel.from_pretrained(config["model_name"])
    tokenizer.bos_token_id = tokenizer.cls_token_id
    tokenizer.eos_token_id = tokenizer.sep_token_id
elif "lama" in config["model_name"]:
    tokenizer = LlamaTokenizerFast.from_pretrained(config["model_name"])
    if config["frozen"]:
        raw_model = LlamaForCausalLM.from_pretrained(
            config["model_name"], device_map="auto"
        )
    else:
        raw_model = LlamaForCausalLM.from_pretrained(config["model_name"])
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    raw_model.resize_token_embeddings(len(tokenizer))
    raw_model.config.pad_token_id = tokenizer.pad_token_id


# DATALOADING

DATABASE_PATH = "data/" + args.dataset + "/"

BASE_MODEL_PATH = "bin"
splits = {}

for file in os.listdir(DATABASE_PATH):
    root_name = file.split(".")[0]
    if root_name in ["train", "valid", "test"]:
        print(os.path.join(DATABASE_PATH, file))
        with open(os.path.join(DATABASE_PATH, file), "rb") as reader:
            splits[root_name] = pickle.load(reader)


try:
    context_size = config["context_size"]
    print(context_size)
except Exception as ex:
    print("Context_size not found!")
    context_size = {"pre": 0, "post": 0}

try:
    only_context_intra = config["only_context_intra"]
except Exception as ex:
    print("Only_context_intra not found!")
    only_context_intra = False
    
print('Configuration:')
pprint(config)

datasets = {}

if "lama" in config["model_name"]:
    prompt_maker = Prompter(
        tokenizer,
        add_spacing=False,
        pre_context=context_size["pre"],
        post_context=context_size["post"],
        only_context_intra=only_context_intra,
    )
else:
    if "add_spacing" in config:
        add_spacing = config['add_spacing']
    else:
        add_spacing = False
    prompt_maker = Prompter(
        tokenizer,
        add_spacing=add_spacing,
        pre_context=context_size["pre"],
        post_context=context_size["post"],
        only_context_intra=only_context_intra,
    )

if config["cls_only"]:
    if config["context"] == "sentence":
        prompt = prompt_maker.events_eos
    elif config["context"] == "paragraph":
        prompt = prompt_maker.events_eos_paragraph
    else:
        print("Prompt not found!")
else:
    prompt = prompt_maker.word_in_sentence

is_skip_vague_train = True
is_skip_vague_dev = True
is_skip_vague_test = True
if args.not_skip_vague:
    if "MATRES" in args.dataset:
        is_skip_vague_train = False
        is_skip_vague_dev = False
        is_skip_vague_test = True
    elif "TIMELINE" in args.dataset:
        is_skip_vague_train = True
        is_skip_vague_dev = True
        is_skip_vague_test = True
    elif "TB-DENSE" in args.dataset:
        is_skip_vague_train = False
        is_skip_vague_dev = False
        is_skip_vague_test = False
    else:
        
        print("NO DATA VAGUE AVAILABLE")
        sys.exit(0)

if config["frozen"]:
    if "bert" in config["model_name"]:
        encode_model = EncoderFrozen(raw_model).to(args.device)
    else:
        encode_model = EncoderFrozen(raw_model)
    
    tmp_model_name = config['model_name'].replace("/", "_") + "_" + args.dataset
    is_there_checkpoint = False
    for x in os.listdir(FROZEN_DUMP_PATH):
        if x == tmp_model_name:
            is_there_checkpoint = True
            
    if is_there_checkpoint:
       with open(os.path.join(FROZEN_DUMP_PATH, tmp_model_name), "rb") as reader:
           datasets  = pickle.load(reader)     
       label_mapping = datasets['train'].dict_rel     
    else:
        print("Train:")
        datasets["train"] = TempDatasetFrozen(
            splits["train"], prompt, encode_model, args.device, skip_vague=is_skip_vague_train
        )
        label_mapping = datasets["train"].dict_rel
        print("Valid:")
        datasets["valid"] = TempDatasetFrozen(
            splits["valid"], prompt, encode_model, args.device, lab_dict=label_mapping, skip_vague=is_skip_vague_dev
        )
        print("Test:")
        datasets["test"] = TempDatasetFrozen(
            splits["test"], prompt, encode_model, args.device, lab_dict=label_mapping, skip_vague=is_skip_vague_test
        )
        with open(os.path.join(FROZEN_DUMP_PATH, tmp_model_name), "wb") as writer:
            pickle.dump(datasets, writer)
    
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        collate_fn=partial(collate_fn_frozen, device=args.device),
        shuffle=True,
    )
    dev_loader = DataLoader(
        datasets["valid"],
        batch_size=32,
        collate_fn=partial(collate_fn_frozen, device=args.device),
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=32,
        collate_fn=partial(collate_fn_frozen, device=args.device),
    )
else:

    datasets["train"] = TempDataset(splits["train"], prompt_function=prompt, skip_vague=is_skip_vague_train)
    label_mapping = datasets["train"].dict_rel
    
    datasets["valid"] = TempDataset(
        splits["valid"], prompt_function=prompt, lab_dict=label_mapping, skip_vague=is_skip_vague_dev
    )
    
    datasets["test"] = TempDataset(
        splits["test"], prompt_function=prompt, lab_dict=label_mapping, skip_vague=is_skip_vague_test
    )
    
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=args.device),
        shuffle=True,
    )

    dev_loader = DataLoader(
        datasets["valid"],
        batch_size=32,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=args.device),
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=32,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=args.device),
    )

comulative_performances = {}
best_outputs = []
model_paths = []
for r in range(config["runs"]):
    # ================ MODELS ==================================

    classfier = FeedForward(config["layers"], len(label_mapping))
    if config["init_weights"]:
        classfier.apply(init_weights)
    
    if config["frozen"]:
        for param in raw_model.parameters():
            param.requires_grad = False
        raw_model.eval()
        model = WrapperClassifier(classifier=classfier)
        model.to(args.device)
    elif config['lora'] != None:
        lora_config = LoraConfig(r=config['lora']['rank'],
                                 lora_alpha=config['lora']['alpha'],  
                                 init_lora_weights="gaussian")
        train_layers1 = 0
        train_layers2 = 0
        train_layers3 = 0

        for param in raw_model.parameters():
            if param.requires_grad:
                train_layers1 += 1
        
        raw_model = get_peft_model(raw_model, lora_config)
        
        for param in raw_model.parameters():
            if param.requires_grad:
                train_layers2 += 1
        
        model = LoRaModel(raw_model, classifier=classfier)
        print('LORA', train_layers1)
        for param in model.parameters():
            if param.requires_grad:
                train_layers3 += 1
        print('LORA2', train_layers2)
        print('LORA3', train_layers3)
        model.to(args.device)
    elif "bert" in config["model_name"]:
        model = BERT_Temp(raw_model, classifier=classfier, text_cls=config["cls_only"], cls_as_context=config['cls_as_context'])
        model.to(args.device)
    
    pprint(model)
  


    # ================ OPTIMIZERS ==================================

    if config["dual_optimizer"]:
        optimizer = optim.AdamW(model.model.parameters(), lr=config["lr1"])
        optimizer_cls = optim.AdamW(model.classifier.parameters(), lr=config["lr2"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config["lr1"])
        optimizer_cls = None

    if not config["frozen"] and  config['lora'] is None:
        print("Linear Scheduler: Active!")
        warmup_steps = int(round(len(datasets["train"]) * config["epochs"] * 0.1))
        training_steps = len(datasets["train"]) * config["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, training_steps
        )
        print("\tWarmup steps", warmup_steps)
    else:
        print("Linear Scheduler: Disabled!")
        scheduler = None
    # ===============================================================

    patience = config["patience"]
    best_loss = math.inf
    best_acc = 0
    best_report = {}
    if args.save_model:
        codexes = [x for x in os.listdir(BASE_MODEL_PATH)]
        codex = (
            datetime.now().isoformat("|")
            + "_"
            + "".join(random.choices(string.ascii_uppercase + string.digits, k=32))
        )
        while codex in codexes:
            codex = (
                datetime.now().isoformat("|")
                + "_"
                + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
            )
        MODEL_PATH = os.path.join(BASE_MODEL_PATH, codex)
        model_paths.append(MODEL_PATH)
    # =============== Train MODEL ==================================

    for i in tqdm(range(0, config["epochs"]), desc="Epochs", unit="epoch"):
        t_loss = train_loop(train_loader, optimizer, optimizer_cls, scheduler, model)
        eval_report, dev_loss, _ = eval_loop(dev_loader, model, label_mapping)
        mean_dev_loss = np.asarray(dev_loss).mean()
        print("="*89)
        print("EPOCH:", i)
        print("Train loss:", np.asarray(t_loss).mean())
        print("Dev loss:", mean_dev_loss)

        pprint(eval_report)
        print("="*89)
        
        if eval_report["accuracy"] > best_acc:
            best_acc = eval_report["accuracy"]
            best_report, _, best_output = eval_loop(test_loader, model, label_mapping)
            if args.save_model:
                torch.save(model.state_dict(), MODEL_PATH)
            patience = config["patience"]
        else:
            patience -= 1
            if patience == 0:
                break

        if best_loss > mean_dev_loss:
            best_loss = mean_dev_loss
            patience = config["patience"]
    # ================== SAVE RESULTS =====================
    for k, v in best_report.items():
        if k not in comulative_performances:
            comulative_performances[k] = []
        comulative_performances[k].append(v)
    if args.save_model:
        comulative_performances["models"] = model_paths
        
    with open("outputs/results.json", "r") as f:
        results = json.loads(f.read())

    for r in results:
        if r["id"] == id_res:
            current_exp = r

    for k, v in comulative_performances.items():
        current_exp[k] = v

    for id_r, r in enumerate(results):
        if r["id"] == id_res:
            results[id_r] = current_exp
    sleep(randint(1,30))
    with lock:
        with open("outputs/results.json", "w") as f:
            f.write(json.dumps(results, indent=4))

    best_outputs.append(best_output)
    # ================== Delete models =====================
    del classfier
    del model


# ================== DUMP OUTPUS =====================
sleep(randint(1,30))

dump_path = os.path.join("outputs", "dump_results.pkl")
if os.path.isfile(dump_path):
    with open(dump_path, "rb") as reader:
        dump = pickle.load(reader)
    dump.append({"id": id_res, "data": best_outputs})
else:
    dump = [{"id": id_res, "data": best_outputs}]
    
lock2 = FileLock(dump_path, timeout=50)
with lock2:
    with open(dump_path, "wb") as f:
        pickle.dump(dump, f, pickle.HIGHEST_PROTOCOL)
