from typing import Dict, List, Tuple
from main import get_all_classes, get_examples, get_uniform
from sandbox.data_structures import Document
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    BertTokenizer,
    BertModel,
    LlamaForCausalLM,
    LlamaTokenizer
)

import pickle
import os

from utils.dataloader import DataRow, PromptPrompter, TempDataset, collate_fn
from utils.model_bertish import BERT_Temp
from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime
import json

from pprint import pprint
from utils.model_llama import Llama_Temp
from utils.support import eval_loop

import argparse

import numpy as np

from tqdm import tqdm
from sklearn.metrics import classification_report

def load_data(
    prompt_maker: PromptPrompter,
    tokenizer,
    splits: dict,
    data_base_path: str,
    config: dict,
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader]:
    datasets = {}
    if config["prompt_type"] == "class":
        if config["prompt"] is None or config["prompt"] == "":
            prompt = prompt_maker.few_shot
        else:
            prompt = prompt_maker.few_shot_prompt

    datasets["valid"] = TempDataset(
        splits["valid"], prompt_function=prompt
    )
    label_mapping = datasets["valid"].dict_rel

    datasets["test"] = TempDataset(
        splits["test"], prompt_function=prompt, lab_dict=label_mapping
    )

    label_mapping["OTHER"] = len(label_mapping)
    print(label_mapping)

    pad_left = "Llama" in config["model_name"]

    # stratified sample of dev set
    valid_subset_path = os.path.join(data_base_path, "valid_subset")
    if os.path.isfile(valid_subset_path):
        print(valid_subset_path)
        with open(valid_subset_path, "rb") as reader:
            datasets["valid"] = pickle.load(reader)
    else:
        datasets["valid"] = TempDataset.stratified_sample(datasets["valid"])
        with open(valid_subset_path, "wb") as f:
            pickle.dump(datasets["valid"], f, pickle.HIGHEST_PROTOCOL)


    dev_loader = DataLoader(
        datasets["valid"],
        batch_size=config["test_batch_size"] if "test_batch_size" in config else 64,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=args.device, pad_left=pad_left),
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=config["test_batch_size"] if "test_batch_size" in config else 64,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=args.device, pad_left=pad_left),
    )
    return dev_loader, test_loader

def get_label_mapping(documents: List[Document]) -> Dict[str, int]:
    label_mapping = {}
    for doc in documents:
        for rel_type in doc.temporal_relations.keys():
            if rel_type != "VAGUE":
                if rel_type not in label_mapping:
                    label_mapping[rel_type] = len(label_mapping)
    label_mapping["OTHER"] = len(label_mapping)
    return label_mapping


def main():
    parser = argparse.ArgumentParser(description="Train Model for Matres")
    parser.add_argument("--config_file", type=str, help="add_path_to_the_config_file")
    parser.add_argument("--device", type=str, help="add_cuda")
    args = parser.parse_args()

    config = json.load(open(args.config_file, "r"))

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
    elif "Llama" in config["model_name"]:
        model_name = config["model_name"]
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        pad_token = "<pad>"
        tokenizer.add_tokens(new_tokens=pad_token, special_tokens=True)
        tokenizer.pad_token = pad_token

        assert config["prompting"]
        model_class = LlamaForCausalLM

        if args.device == "cpu":
            raw_model = model_class.from_pretrained(model_name).to("cpu")
        elif args.device is not None and "cuda" in args.device:
            raw_model = model_class.from_pretrained(model_name).to(args.device)
        else:
            raw_model = model_class.from_pretrained(model_name, device_map="auto")
        raw_model.resize_token_embeddings(len(tokenizer))
    raw_model.eval()

    # DATALOADING
    data_base_path = "data/MATRES"
    splits = {}

    for file in os.listdir(data_base_path):
        root_name = file.split(".")[0]
        if root_name in ["train", "valid", "test"]:
            print(os.path.join(data_base_path, file))
            with open(os.path.join(data_base_path, file), "rb") as reader:
                splits[root_name] = pickle.load(reader)

    datasets = {}
    all_classes = get_all_classes(splits)

    examples_list = []
    save_examples_file = "./data/few_shot_examples_list_100.pkl"
    few_shots = 10
    samples = 100
    if os.path.isfile(save_examples_file):
        print("Load few-shot examples")
        with open(save_examples_file, "rb") as f:
            examples_list = pickle.load(f)
    else:
        for _ in range(samples):
            examples = get_examples(splits["train"], few_shots)
            examples = get_uniform(examples, few_shots, all_classes)
            examples_list.append(examples)
        print("Save few-shot examples")
        with open(save_examples_file, "wb") as f:
            pickle.dump(examples_list, f, pickle.HIGHEST_PROTOCOL)

    label_mapping = get_label_mapping(splits["valid"])

    # ================ MODELS ==================================

    if "Llama" in config["model_name"]:
        max_new_tokens = 10
        if "MATRES" in data_base_path:
            max_new_tokens = 5
        model = Llama_Temp(
            raw_model,
            label_mapping,
            tokenizer,
            # classifier=classfier,
            classifier=None,
            text_cls=config["cls_only"],
            prompting=config["prompting"],
            max_new_tokens=max_new_tokens
        )
    else:
        model = BERT_Temp(raw_model, classifier=classfier, text_cls=config["cls_only"])
    pprint(model)
    if "Llama" not in config["model_name"] and args.device is not None and (args.device == "cpu" or "cuda" in args.device):
        model.to(args.device)

    best_examples = None
    best_valid_perf = 0
    for i, examples in enumerate(examples_list):
        for doc, ev1, ev2, label in examples:
            print("-"*100)
            print(doc.text[ev1.sent_id], doc.text[ev2.sent_id])
            print(ev1.token)
            print(ev2.token)
            print(label)
        prompt_maker = PromptPrompter(
            config["prompt"],
            all_classes,
            tokenizer,
            truncate_limit=tokenizer.model_max_length,
            examples=examples,
            highlight_events=config["highlight_events"],
        )
        dev_loader, test_loader = load_data(prompt_maker, tokenizer, splits, data_base_path, config, args)

        # =============== eval ==================================


        eval_report, dev_loss, _ = eval_loop(dev_loader, model, label_mapping)

        print("Example set:", i)
        pprint(eval_report)

        metric = eval_report["accuracy"] if "accuracy" in eval_report.keys() else eval_report["micro avg"]["f1-score"]
        if metric > best_valid_perf:
            best_valid_perf = metric
            best_examples = examples
            best_valid_report = eval_report

    # =============== test best ==================================
    prompt_maker = PromptPrompter(
        config["prompt"],
        all_classes,
        tokenizer,
        truncate_limit=tokenizer.model_max_length,
        # use best examples
        examples=best_examples,
        highlight_events=config["highlight_events"],
    )
    dev_loader, test_loader = load_data(prompt_maker, tokenizer, splits, data_base_path, config, args)

    comulative_performances = {}
    best_outputs = []
    best_report, _, best_output = eval_loop(test_loader, model, label_mapping)

    for k, v in best_report.items():
        if k not in comulative_performances:
            comulative_performances[k] = []
        comulative_performances[k].append(v)
    best_outputs.append(best_output)
    del model


    #================== SAVE RESULTS =====================

    results_file = "outputs/results_optim_few_shot.json"
    if not os.path.isfile(results_file):
        with open(results_file, "w") as f:
            f.write(json.dumps([{}], indent=4))

    with open(results_file, "r") as f:
        results = json.loads(f.read())


    id_t = len(results)

    comulative_performances["id"] = id_t
    comulative_performances["created_at"] = datetime.now().isoformat("|")
    comulative_performances["config"] = config

    results.append(comulative_performances)

    with open(results_file, "w") as f:
        f.write(json.dumps(results, indent=4))

    dump_path = os.path.join("outputs", "dump_results_optim_few_shot.pkl")
    if os.path.isfile(dump_path):
        with open(dump_path, "rb") as reader:
            dump = pickle.load(reader)
        dump.append({"id": id_t, "data": best_outputs})
    else:
        dump = [{"id": id_t, "data": best_outputs}]

    with open(dump_path, "wb") as f:
        pickle.dump(dump, f, pickle.HIGHEST_PROTOCOL)

    # save examples that resulted in best performance on dev
    best_examples_path = "outputs/best_examples.pkl"
    with open(best_examples_path, "wb") as f:
        pickle.dump(best_examples, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
