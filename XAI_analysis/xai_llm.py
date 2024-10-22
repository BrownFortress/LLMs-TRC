from pathlib import Path
import shutil
import string
from typing import Any, Dict, List, Optional, Tuple

from torch import optim
from main_prompting import get_all_classes, get_uniform
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, PeftModel

import pickle
import os

from utils.dataloader import PromptPrompter, TempDataset, TempQuestionsDataset, TextPromptPrompter, collate_fn, fine_tune_collate_fn
from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime
import json

from pprint import pprint
from utils.model_llama import Llama_Temp
import argparse

import random
from filelock import FileLock
from time import sleep
from tqdm import tqdm
import numpy as np
import torch


from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    KernelShap,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
    ProductBaselines,
)

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

def init_model(model_name: str, tokenizer: AutoTokenizer, args: argparse.Namespace, config: Dict[str, Any], update_tokenizer: bool = False, add_lora_if_avail: bool = True, force_cpu=False) -> AutoModelForCausalLM:
    # Huggingface models
    if args.device == "cpu" or force_cpu:
        raw_model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
    elif args.device is not None and "cuda" in args.device:
        print("I'm on device", args.device)
        raw_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(args.device)
    else: 
        print("I'm on device Das Auto")
        raw_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    if "Llama-2-70b" not in model_name and "Mixtral-8x7B" not in model_name:
        pad_token = "<pad>"
        if update_tokenizer:
            tokenizer.add_tokens(new_tokens=pad_token, special_tokens=True)
            tokenizer.pad_token = tokenizer.eos_token
        # resize (added <pad> token)
        raw_model.resize_token_embeddings(len(tokenizer))
    else:
        if update_tokenizer:
            tokenizer.pad_token = tokenizer.eos_token

    if args.device is not None and (args.device == "cpu" or "cuda" in args.device):
        raw_model.to(args.device)

    return raw_model

def main():
    parser = argparse.ArgumentParser(description="XAI LLMs")
    parser.add_argument("--config_file", type=str, help="add_path_to_the_config_file", default="configs/prompting/llama2/7b/few_shot_3_ev_in_q.json")
    parser.add_argument("--device", type=str, help="add_cuda")
    parser.add_argument("--dataset", type=str, help="add a dataset", default="MATRES", required=True, choices=["MATRES", "TIMELINE", "TB-DENSE"])
    parser.add_argument("--out_dir", type=str, help="", default="outputs")
    parser.add_argument("--model_path", type=str, help="")

    args = parser.parse_args()

        
    print(f"Loading config file '{args.config_file}' ...")
    config = json.load(open(args.config_file, "r"))

    config['dataset_name'] = args.dataset
    
    if args.model_path is not None:
        config['model_path'] = args.model_path
    

    # ================= Init Model ============================
    print(f"Load model '{config['model_name']}' ...")
    model_name = config["model_name"]
    assert config["prompting"]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


    # os.system("nvidia-smi")
    raw_model = init_model(model_name, tokenizer, args, config, update_tokenizer=True, force_cpu=True)
    print("PAD TOKEN: ", tokenizer.pad_token)

    raw_model.eval()

    # os.system("nvidia-smi")

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
    # if "fine_tune" not in config or not config["fine_tune"]:
    if config["few_shots"] > 0:
        # assert config["few_shots"] > 0
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
    n_runs = 1
    if examples_list is not None:
        n_runs = len(examples_list)
    if "runs" in config and config["runs"] is not None:
        # assert config["runs"] <= len(examples_list)
        n_runs = config["runs"]
        
    n_runs = 1
    for run in range(n_runs):
        print("*"*100)
        print(f"Run: {run+1}/{n_runs}")
        examples = None
        if examples_list is not None:
            examples = examples_list[run]
            
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

        skip_vague = args.dataset != "TB-DENSE"
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

        # if args.dataset == "TB-DENSE":
        #     del label_mapping["VAGUE"]
        label_mapping["OTHER"] = len(label_mapping)
        print(label_mapping)

        pad_left = True
        questions_seq = config["prompt_type"] == "questions_sequential"
        # TODO also include davinci-002
        # text_emb = model_name in API_MODELS and model_name != "davinci-002"
        text_emb = False

    
        test_loader = DataLoader(
            datasets["test"],
            batch_size=1,
            collate_fn=partial(collate_fn, tokenizer=tokenizer, device=args.device, pad_left=pad_left, questions_seq=questions_seq, text_emb=text_emb),
        )
        if "model_path" in config:
            save_folder = "./bin"
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, config['model_path'])
            
            # load base model
            base_model = init_model(model_name, tokenizer, args, config, update_tokenizer=False, add_lora_if_avail=False)
            # load lora adapters
            raw_model = PeftModel.from_pretrained(base_model, save_path)
            print(raw_model.print_trainable_parameters())

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

        model.eval()
        rev = {v:k for k, v in label_mapping.items()}

        # fa = FeatureAblation(model.model)
        #  fa = ShapleyValues(model.model)
        fa = KernelShap(model.model)
        llm_attr = LLMAttribution(fa, tokenizer)
        # fa = LayerIntegratedGradients(model.model, model.model.model.embed_tokens)
        # fa = ShapleyValues(model.model)
        # llm_attr = LLMGradientAttribution(fa, tokenizer)
        correct_examples = []
        wrong_examples = []
        oov = []
        for sample in tqdm(test_loader, desc="Evaluating", unit="batch"):
            questions = None
            if rev[sample['labels'][0].item()] == "VAGUE":
                continue
            if "questions_ids" in sample:
                # questions_sequential
                questions=sample["questions_ids"]
            if  rev[sample['labels'][0].item()] != "VAGUE":
                with torch.no_grad():
                    steps, label = model(
                        sample['input_ids'],
                        sample['attention_mask'],
                        sample['labels'],
                        sample['masks'],
                        sample['complementary_mask'],
                        sample['to_duplicate'],
                        questions=questions,
                        unroll=True, 
                        device=args.device
                    )
                print(label, steps)
       
                context = tokenizer.decode(sample['input_ids'][0])
                context = TextTokenInput(context, tokenizer, skip_tokens=[1])
            
                target = rev[sample['labels'][0].item()]
            
                attr_res = llm_attr.attribute(context, target=target, n_samples=300)
                if steps is not None:
                    if label == rev[sample['labels'][0].item()]:      
                        correct_examples.append(attr_res) 
                        wrong_examples.append(None)
                        oov.append(None)
                    else:
                        wrong_examples.append(attr_res)
                        correct_examples.append(None) 
                        oov.append(None)
                else:
                    correct_examples.append(None)
                    wrong_examples.append(None)
                    oov.append(attr_res)
           
    res = {"good":correct_examples, "bad":wrong_examples, "wierd":oov}            
    with open(args.dataset+"_finetuned_kernelShap.pkl", "wb") as file:
        file.write(pickle.dumps(res))

if __name__ == "__main__":
    main()
