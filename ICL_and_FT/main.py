import shutil
import string
from typing import Any, Dict, List, Optional, Tuple

from torch import optim
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import pickle
import os

from utils.dataloader import PromptPrompter, TempDataset, TempQuestionsDataset, TextPromptPrompter, collate_fn, fine_tune_collate_fn
from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime
import json

from pprint import pprint
from utils.model_llama import Llama_Temp
from utils.support import eval_loop, train_loop_prompt

import argparse

import random
from filelock import FileLock
from time import sleep
from tqdm import tqdm
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from utils.chatgpt_model import ChatGPTModel

API_MODELS = ["davinci-002", "gpt-3.5-turbo-0125"]

def fine_tune(
    save_path: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    datasets: Dict[str, TempDataset],
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    label_mapping: Dict[str, int],
    rel_type_order: List[str],
    config: Dict[str, Any],
    args: argparse.Namespace,
    max_new_tokens: int = 5,
    skip_vague: bool = False,
    test: bool = False
):

    # ================ OPTIMIZERS ==================================
    optimizer = optim.AdamW(model.parameters(), lr=config["lr1"])

    if ("frozen" not in config or not config["frozen"]) and ("lora" not in config or config['lora'] is None):
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

    # ================ TRAIN  ==================================
    model.train()
    best_acc = -1
    patience = config["patience"]
    for i in range(0, config["epochs"]):
        t_loss = train_loop_prompt(train_loader, optimizer, scheduler, model, tokenizer, config, i, test=test)
        print(f"EPOCH {i}: Loss array: {t_loss}")
        eval_model = get_eval_model(model, tokenizer, label_mapping, rel_type_order, config, max_new_tokens=max_new_tokens)
        print(eval_model)
        eval_report, dev_loss, eval_output, eval_output_str = eval_loop(dev_loader, eval_model, label_mapping, rel_type_order, args.dataset, skip_vague, test=test)
        del eval_model
        print("EPOCH:", i)
        print("Train loss:", np.asarray(t_loss).mean())

        pprint(eval_report)

        eval_acc = -1
        if "accuracy" in eval_report:
            eval_acc = eval_report["accuracy"]
        elif "micro avg" in eval_report:
            eval_acc = eval_report["micro avg"]["f1-score"]
        print(f"EVAL ACC: {eval_acc}")

        if eval_acc > best_acc:
            best_acc = eval_acc
            print(f"Saving best model to '{save_path}' ...")
            model.save_pretrained(save_path)
            patience = config["patience"]
        else:
            patience -= 1
            if patience == 0:
                break
        if test:
            break

def get_eval_model(
    raw_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    label_mapping: Dict[str, int],
    rel_type_order: List[str],
    config: Dict[str, Any],
    max_new_tokens: int = 5
) -> Llama_Temp:
    model = Llama_Temp(
        raw_model,
        label_mapping,
        tokenizer,
        classifier=None,
        text_cls=config["cls_only"],
        prompting=config["prompting"],
        max_new_tokens=max_new_tokens,
        prompt_type=config["prompt_type"],
        sys_prompt_type=config["sys_prompt_type"] if "sys_prompt_type" in config else None,
        rel_type_order=rel_type_order,
        save_logits=config["save_logits"] if "save_logits" in config else False
    )
    return model

def init_model(
        model_name: str,
        tokenizer: AutoTokenizer,
        args: argparse.Namespace,
        config: Dict[str, Any],
        update_tokenizer: bool = False,
        add_lora_if_avail: bool = True,
        device: Optional[str] = None
) -> AutoModelForCausalLM:
    if device is None:
        device = args.device

    if model_name in API_MODELS:
        # OpenAI API models
        with open("key.json", "r") as f:
            key_file = json.load(f)
        raw_model = ChatGPTModel(key_file["key"], tokenizer, model_name)
        if update_tokenizer:
            pad_token = "<pad>"
            tokenizer.add_tokens(new_tokens=pad_token, special_tokens=True)
            tokenizer.pad_token = pad_token
    else:
        # Huggingface models
        dtype = "auto"
        if "lora" in config and config['lora'] != None:
            dtype = torch.bfloat16

        if device == "cpu":
            raw_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cpu")
        elif device is not None and "cuda" in device:
            raw_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        else:
            raw_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

        if "Llama-2-70b" not in model_name and "Mixtral-8x7B" not in model_name:
            pad_token = "<pad>"
            if update_tokenizer:
                tokenizer.add_tokens(new_tokens=pad_token, special_tokens=True)
                tokenizer.pad_token = pad_token
            # resize (added <pad> token)
            raw_model.resize_token_embeddings(len(tokenizer))
        else:
            if update_tokenizer:
                tokenizer.pad_token = tokenizer.eos_token

        # LoRA
        if add_lora_if_avail and "lora" in config and config['lora'] != None:
            lora_config = LoraConfig(r=config['lora']['rank'],
                                     lora_alpha=config['lora']['alpha'],
                                     init_lora_weights="gaussian")
            train_layers1 = 0
            train_layers2 = 0

            for param in raw_model.parameters():
                if param.requires_grad:
                    train_layers1 += 1

            raw_model = get_peft_model(raw_model, lora_config)

            for param in raw_model.parameters():
                if param.requires_grad:
                    train_layers2 += 1

            print('LORA', train_layers1)
            print('LORA2', train_layers2)

    if device is not None and (device == "cpu" or "cuda" in device):
        raw_model.to(device)

    return raw_model


def get_all_classes(splits: Dict[str, List[Document]]) -> List[str]:
    all_classes = []
    for split in splits.values():
        for doc in split:
            for rel_type in doc.temporal_relations.keys():
                if rel_type != "VAGUE":
                    if rel_type not in all_classes:
                        all_classes.append(rel_type)
    return all_classes


def get_examples(train_docs: List[Document], few_shots: int) -> List[Tuple[Document, Event, Event, str]]:
    examples_by_class = {}
    for doc in train_docs:
        for rel_type, relations in doc.temporal_relations.items():
            if rel_type == "VAGUE":
                continue
            for rel in relations:
                if rel_type not in examples_by_class:
                    examples_by_class[rel_type] = []
                examples_by_class[rel_type].append((doc, rel.event1, rel.event2, rel_type))
    id_to_rel = {
        i: rel for i, rel in enumerate(examples_by_class.keys())
    }
    n_classes = len(id_to_rel)

    class_ids_list = list(range(n_classes))
    if n_classes < few_shots:
        # each class taken the same number of times
        class_samples = class_ids_list * (few_shots // n_classes)
        # the rest is sampled uniformly from the classes
        rest = few_shots % n_classes
        class_samples.extend(random.sample(class_ids_list, rest))
    else:
        # samples classes uniformly
        class_samples = random.sample(class_ids_list, few_shots)

    assert len(class_samples) == few_shots

    class_samples_counts = {}
    for rel_id in class_samples:
        rel_type = id_to_rel[rel_id]
        if rel_type not in class_samples_counts:
            class_samples_counts[rel_type] = 0
        class_samples_counts[rel_type] += 1

    examples = []
    for rel_type, n_samples in class_samples_counts.items():
        examples.extend(random.sample(examples_by_class[rel_type], n_samples))

    assert len(examples) == few_shots

    return examples

def get_uniform(examples: List[Tuple[Document, Event, Event, str]], few_shots: int, all_classes: List[str]):
    if len(examples) == few_shots:
        per_class_counts = {rel_type: 0 for rel_type in all_classes}
        for ex in examples:
            ex_rel_type = ex[3]
            per_class_counts[ex_rel_type] += 1
    else:
        per_class_counts = {rel_type: (few_shots//len(all_classes)) for rel_type in all_classes}
        rest = few_shots % len(all_classes)
        for i, rel_type in enumerate(list(per_class_counts.keys())):
            if i >= rest:
                break
            per_class_counts[rel_type] += 1
    assert sum(per_class_counts.values()) == few_shots
    per_class_counts_copy = copy.deepcopy(per_class_counts)
    ex_per_class = {rel_type: [] for rel_type in all_classes}
    for ex in examples:
        ex_rel_type = ex[3]
        if per_class_counts_copy[ex_rel_type] <= 0:
            continue
        per_class_counts_copy[ex_rel_type] -= 1
        ex_per_class[ex_rel_type].append(ex)

    assert sum([len(exs) for exs in ex_per_class.values()]) == few_shots
    out_examples = []
    for rel_type, exs in ex_per_class.items():
        out_examples.extend(exs[:per_class_counts[rel_type]])

    assert len(out_examples) == few_shots
    return out_examples

def main():
    parser = argparse.ArgumentParser(description="Train Model for Matres")
    parser.add_argument("--config_file", type=str, help="add_path_to_the_config_file")
    parser.add_argument("--device", type=str, help="add_cuda")
    parser.add_argument("--dataset", type=str, help="add a dataset", required=True, choices=["MATRES", "TIMELINE", "MATRES_NEW", "TB-DENSE"])
    parser.add_argument("--out_dir", type=str, help="", default="outputs")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print("TEST")


    print(f"Loading config file '{args.config_file}' ...")
    config = json.load(open(args.config_file, "r"))

    config['dataset_name'] = args.dataset

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    res_file = f"{out_dir}/results.json"
    print(f"Out dir: '{out_dir}'")

    sleep_time = random.sample(list(range(0,20)), k=1)[0]
    print(f"Random sleep: {sleep_time}s")
    sleep(sleep_time)

    # ================= Init Save results ====================
    print(f"Init results file...")
    if not os.path.isfile(res_file):
        with FileLock(f"{res_file}.lock"):
            with open(res_file, "w") as f:
                f.write(json.dumps([], indent=4))
    with FileLock(f"{res_file}.lock"):
        with open(res_file, "r") as f:
            results = json.loads(f.read())
    id_res = (
        datetime.now().isoformat("|")
        + "_"
        + "".join(random.choices(string.ascii_uppercase + string.digits, k=32))
    )

    new_row = {"id": id_res, "created_at": datetime.now().isoformat("|"), "config": config}
    results.append(new_row)
    with FileLock(f"{res_file}.lock"):
        with open(res_file, "w") as f:
            f.write(json.dumps(results, indent=4))


    # ================= Init Model ============================
    print(f"Load model '{config['model_name']}' ...")
    model_name = config["model_name"]
    assert config["prompting"]
    if model_name not in API_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


    os.system("nvidia-smi")
    raw_model = init_model(model_name, tokenizer, args, config, update_tokenizer=True)
    print("DEVICE: ", raw_model.device)
    print("DTYPE: ", raw_model.dtype)
    print("PAD TOKEN: ", tokenizer.pad_token)

    raw_model.eval()

    os.system("nvidia-smi")

    # DATALOADING
    print(f"Load data...")
    data_base_path = f"data/{args.dataset}"
    splits = {}

    for file in os.listdir(data_base_path):
        root_name = file.split(".")[0]
        if root_name in ["train", "valid", "test"]:
            print(os.path.join(data_base_path, file))
            with open(os.path.join(data_base_path, file), "rb") as reader:
                splits[root_name] = pickle.load(reader)

    datasets = {}

    if args.dataset == "TB-DENSE":
        rel_type_order = get_all_classes(splits)
    else:
        rel_type_order = ["AFTER", "BEFORE", "EQUAL"]
    if "rel_type_order" in config and config["rel_type_order"] is not None:
        rel_type_order = config["rel_type_order"]
    else:
        config["rel_type_order"] = rel_type_order
    print(rel_type_order)

    examples_list = None
    if config["few_shots"] > 0:
        save_examples_file = os.path.join(f"./saved_examples/{args.dataset}", f"samples_{config['few_shots']}_shot.pkl")
        print(f"Load few-shot examples from {save_examples_file}")
        with open(save_examples_file, "rb") as f:
            examples_list = pickle.load(f)
        # order depends on all_classes
        examples_list = [get_uniform(examples, config["few_shots"], rel_type_order) for examples in examples_list]
        for examples in examples_list:
            print("="*100)
            for doc, ev1, ev2, label in examples:
                print("-"*100)
                print(doc.text[ev1.sent_id], doc.text[ev2.sent_id])
                print(ev1.token)
                print(ev2.token)
                print(label)

    questions = None
    if "questions" in config["prompt_type"]:
        if args.dataset == "TB-DENSE":
            questions = {
                "BEFORE": "Does event1 happen before event2?",
                "AFTER": "Does event1 happen after event2?",
                "SIMULTANEOUS": "Does event1 happen at the same time as event2?",
                "INCLUDES": "Does event1 temporally include event2?",
                "IS_INCLUDED": "Is event1 temporally included in event2?",
            }
        else:
            questions = {
                "BEFORE": "Does event1 happen before event2?",
                "AFTER": "Does event1 happen after event2?",
                "EQUAL": "Does event1 happen at the same time as event2?"
            }
        config["questions"] = questions

    max_new_tokens = 5
    if args.dataset == "TB-DENSE" and "question" not in config["prompt_type"]:
        max_new_tokens = 10

    # select prompter class
    prompter_class = PromptPrompter
    # if model_name in API_MODELS and "prompt" in config and config["prompt"] != "":
    if model_name in API_MODELS:
        # openai api models
        prompter_class = TextPromptPrompter

    comulative_performances = {}
    best_outputs = []
    best_out_str_list = []
    n_runs = 1
    if examples_list is not None:
        n_runs = len(examples_list)
    if "runs" in config and config["runs"] is not None:
        n_runs = config["runs"]
    for run in range(n_runs):
        print("*"*100)
        print(f"Run: {run+1}/{n_runs}")
        examples = None
        if examples_list is not None:
            examples = examples_list[run]
            for doc, ev1, ev2, label in examples:
                print("-"*100)
                print(doc.text[ev1.sent_id], doc.text[ev2.sent_id])
                print(ev1.token)
                print(ev2.token)
                print(label)
        prompt_maker = prompter_class(
            config["prompt"],
            rel_type_order,
            tokenizer,
            truncate_limit=tokenizer.model_max_length,
            examples=examples,
            rel_type_order=rel_type_order,
            highlight_events=config["highlight_events"],
            questions=questions,
            prompt_type=config["prompt_type"],
            ev_in_question=("ev_in_question" in config and config["ev_in_question"])
        )

        ds_class = TempDataset
        if config["prompt_type"] == "class":
            if config["prompt"] is None or config["prompt"] == "":
                prompt = prompt_maker.few_shot
            else:
                prompt = prompt_maker.few_shot_prompt
            ds_class = TempDataset
        elif config["prompt_type"] in ["questions_single", "questions_all"]:
            if config["prompt"] is None or config["prompt"] == "":
                prompt = prompt_maker.few_shot_questions
            else:
                prompt = prompt_maker.few_shot_questions_prompt
            ds_class = TempQuestionsDataset
        elif config["prompt_type"] == "questions_sequential":
            if config["prompt"] is None or config["prompt"] == "":
                prompt = prompt_maker.few_shot_questions_seq
            else:
                prompt = prompt_maker.few_shot_questions_seq_prompt
            ds_class = TempDataset

        # skip vague by default
        skip_vague = True
        if "skip_vague" in config:
            # skip vague depending on config
            skip_vague = config["skip_vague"]
        print(f"SKIP VAGUE: {skip_vague}")

        datasets["train"] = ds_class(
            splits["train"], prompt_function=prompt, skip_vague=skip_vague
        )
        label_mapping = datasets["train"].dict_rel

        datasets["valid"] = ds_class(
            splits["valid"], prompt_function=prompt, lab_dict=label_mapping, skip_vague=skip_vague
        )

        datasets["test"] = ds_class(
            splits["test"], prompt_function=prompt, lab_dict=label_mapping, skip_vague=skip_vague
        )
        for ds_split in datasets.values():
            assert "VAGUE" not in [x.label for x in ds_split]

        label_mapping["OTHER"] = len(label_mapping)
        print(f"MAPPING: {label_mapping}")

        pad_left = True
        questions_seq = config["prompt_type"] == "questions_sequential"
        text_emb = model_name in API_MODELS

        if "fine_tune" in config and config["fine_tune"]:
            train_loader = DataLoader(
                datasets["train"],
                batch_size=config["batch_size"],
                collate_fn=partial(fine_tune_collate_fn, tokenizer=tokenizer, device=args.device, rel_type_order=rel_type_order, prompt_type=config["prompt_type"]),
            )
        dev_loader = DataLoader(
            datasets["valid"],
            batch_size=config["test_batch_size"] if "test_batch_size" in config else 64,
            collate_fn=partial(collate_fn, tokenizer=tokenizer, device=args.device, pad_left=pad_left, questions_seq=questions_seq, text_emb=text_emb),
        )
        test_loader = DataLoader(
            datasets["test"],
            batch_size=config["test_batch_size"] if "test_batch_size" in config else 64,
            collate_fn=partial(collate_fn, tokenizer=tokenizer, device=args.device, pad_left=pad_left, questions_seq=questions_seq, text_emb=text_emb),
        )


        # =============== Train MODEL =================================
        if "fine_tune" in config and config["fine_tune"]:
            save_folder = "./bin"
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"{id_res}_best_model")

            fine_tune(save_path, raw_model, tokenizer, datasets, train_loader, dev_loader, test_loader, label_mapping, rel_type_order, config, args, max_new_tokens=max_new_tokens, skip_vague=skip_vague, test=args.test)

            # delete old model
            del raw_model

            # load best weights
            print(f"Loading best model from '{save_path}' ...")
            if "lora" not in config or config["lora"] is None:
                # load from save_path
                raw_model = init_model(save_path, tokenizer, args, config, update_tokenizer=False, add_lora_if_avail=False)
            else:
                # load base model
                base_model = init_model(model_name, tokenizer, args, config, update_tokenizer=False, add_lora_if_avail=False, device="cpu")
                # load lora adapters
                raw_model = PeftModel.from_pretrained(base_model, save_path, device_map="auto")

            print("DEVICE: ", raw_model.device)
            print("DTYPE: ", raw_model.dtype)

            if "lora" not in config or config["lora"] is None:
                # delete unless just lora adapters
                print(f"Delete save '{save_path}' ...")
                try:
                    shutil.rmtree(save_path)
                except:
                    print(f"Failed to delete '{save_path}'")


        # =============== Test MODEL ==================================
        raw_model.eval()
        model = get_eval_model(raw_model, tokenizer, label_mapping, rel_type_order, config, max_new_tokens=max_new_tokens)
        pprint(model)
        if args.device is not None and (args.device == "cpu" or "cuda" in args.device):
            model.to(args.device)

        best_report = {}

        best_report, _, best_output, best_output_str = eval_loop(test_loader, model, label_mapping, rel_type_order, args.dataset, skip_vague, test=args.test)
        best_outputs.append(best_output)
        best_out_str_list.append(best_output_str)

        # ================== SAVE RESULTS =====================
        for k, v in best_report.items():
            if k not in comulative_performances:
                comulative_performances[k] = []
            comulative_performances[k].append(v)

        with FileLock(f"{res_file}.lock"):
            with open(res_file, "r") as f:
                results = json.loads(f.read())

        for r in results:
            if r["id"] == id_res:
                current_exp = r

        for k, v in comulative_performances.items():
            current_exp[k] = v

        for id_r, r in enumerate(results):
            if r["id"] == id_res:
                results[id_r] = current_exp

        with FileLock(f"{res_file}.lock"):
            with open(res_file, "w") as f:
                f.write(json.dumps(results, indent=4))

        # ================== Delete models =====================
        del model


    # ================== DUMP OUTPUS =====================
    dump_path = os.path.join(out_dir, "dump_results.pkl")
    if os.path.isfile(dump_path):
        with FileLock(f"{dump_path}.lock"):
            with open(dump_path, "rb") as reader:
                dump = pickle.load(reader)
        dump.append({"id": id_res, "data": best_outputs, "out_str": best_out_str_list})
    else:
        dump = [{"id": id_res, "data": best_outputs, "out_str": best_out_str_list}]

    with FileLock(f"{dump_path}.lock"):
        with open(dump_path, "wb") as f:
            pickle.dump(dump, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
