from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from utils.dataloader import B_INST, E_INST, DataRow
from utils.model_llama import Llama_Temp


def train_loop(data, optimizer, scheduler, model):
    model.train()
    loss_array = []
    for sample in tqdm(data):
        loss, _ = model(sample['input_ids'], sample['attention_mask'], sample['labels'], sample['masks'], sample['complementary_mask'], sample['to_duplicate'])

        loss_array.append(loss.item())
        optimizer.zero_grad()  # Zeroing the gradient
        # optimizer_cls.zero_grad()
        loss.backward()  # Compute the gradient, deleting the computational graph
        optimizer.step()
        # optimizer_cls.step()
        scheduler.step()
    return loss_array

def train_fwd(model, tokenizer, config, sample, rev_mapping):
    q_ids = sample["input_ids"]
    if config["prompt_type"] == "class":
        labels = [rev_mapping[lab] for lab in sample["labels"].tolist()]
        # a_ids = torch.tensor(tokenizer(labels, add_special_tokens=False).input_ids)
    elif config["prompt_type"] in ["questions_single", "questions_all"]:
        row = sample["data_rows"][0]
        comp_rel_type = row.label
        rel_type, q_rel_type = comp_rel_type.split("__")
        if rel_type == q_rel_type:
            labels = ["YES"]
        else:
            labels = ["NO"]

    a_ids = torch.tensor(tokenizer(labels, add_special_tokens=False).input_ids)
    input_ids = torch.cat([q_ids.to(model.device), a_ids.to(model.device)], dim=1)
    attention_mask = torch.ones(input_ids.shape)
    labels = input_ids.clone()
    print(tokenizer.batch_decode(input_ids)[0])
    print("-"*100)

    out = model(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        labels=labels.to(model.device),
    )
    return out

def train_fwd_seq(model, tokenizer, config, sample, rev_mapping, rel_type_order):
    outputs = []
    assert "questions_ids" in sample
    questions = sample["questions_ids"][0]
    rel_types = [rev_mapping[lab] for lab in sample["labels"].tolist()]
    input_ids = sample["input_ids"]

    sys_prompt_type = None
    if "sys_prompt_type" in config:
        sys_prompt_type = config["sys_prompt_type"]

    q_a_embs = [] # will contain q1, a1, q2, a2, ...
    labels = []
    # input_ids + q1
    # input_ids + q1 + a1 + q2
    # input_ids + q1 + a1 + q2 + a2 + q3
    eos_bos_emb  = torch.LongTensor([tokenizer.eos_token_id, tokenizer.bos_token_id])
    b_inst_emb = tokenizer(B_INST, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    e_inst_emb = tokenizer(E_INST + " ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    for i, q_rel_type in enumerate(rel_type_order):
        question_emb = questions[q_rel_type]
        if sys_prompt_type is not None:
            # chat prompt
            if sys_prompt_type == "single_inst":
                q_a_embs.append(question_emb.unsqueeze(0))
                q_a_embs.append(e_inst_emb.unsqueeze(0))
            elif sys_prompt_type == "question_inst":
                if i == 0:
                    question_emb = torch.cat([question_emb.to(model.device), e_inst_emb.to(model.device)], dim=0).type(torch.LongTensor)
                else:
                    question_emb = torch.cat([eos_bos_emb.to(model.device), b_inst_emb.to(model.device), question_emb.to(model.device), e_inst_emb.to(model.device)], dim=0).type(torch.LongTensor)
                q_a_embs.append(question_emb.unsqueeze(0))
        else:
            # not chat
            q_a_embs.append(question_emb.unsqueeze(0))

        # append Ground Truth answer
        if rel_types[0] == q_rel_type:
            answ_str = ["YES"]
        else:
            answ_str = ["NO"]
        answ_emb = torch.tensor(tokenizer(answ_str, add_special_tokens=False).input_ids)
        q_a_embs.append(answ_emb)

        print("input ids:", input_ids.device)
        print("q_a_embs:", [e.device for e in q_a_embs])
        input_ids_with_question = torch.cat([input_ids.to(model.device)] + [e.to(model.device) for e in q_a_embs], dim=1).type(torch.LongTensor)
        attention_mask_with_question = torch.ones(input_ids_with_question.shape)
        print("rel_type: ", rel_types[0])
        print(tokenizer.batch_decode(input_ids_with_question)[0])
        print("-"*100)
        labels = input_ids_with_question.clone()
        out = model(
            input_ids=input_ids_with_question.to(model.device),
            attention_mask=attention_mask_with_question.to(model.device),
            labels=labels.to(model.device),
        )
        outputs.append(out)

        if sys_prompt_type is not None:
            # chat prompt
            if sys_prompt_type == "single_inst":
                # remove [/INST]
                q_a_embs.pop()
    return outputs



def train_loop_prompt(data, optimizer, scheduler, model, tokenizer, config: dict, epoch: int):
    model.train()
    loss_array = []
    for sample in tqdm(data, desc=f"Epoch [{epoch+1}/{config['epochs']}]", unit="batch"):
        input_ids = sample["input_ids"].to(model.device)
        attention_mask = sample["attention_mask"].to(model.device)
        labels = input_ids.clone()
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = out.loss

        loss_array.append(loss.item())
        if optimizer is not None:
            optimizer.zero_grad()  # Zeroing the gradient
        loss.backward()  # Compute the gradient, deleting the computational graph
        if optimizer is not None:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return loss_array

def determine_preds(
    gt_rels: List[str],
    pred_rels: List[int],
    pred_logits: List[Tuple[torch.Tensor]],
    ev_ids: List[Tuple[str, str]],
    data_rows: List[DataRow],
    mapping: Dict[str, int]
):
    grouped = {}
    out_gt_rels = []
    out_pred_rels = []
    for gt_rel, pred_rel, logits, id, row in zip(gt_rels, pred_rels, pred_logits, ev_ids, data_rows):
        if id not in grouped:
            grouped[id] = {}
        # grouped[id][gt_rel] = pred_rel
        if "preds" not in grouped[id]:
            grouped[id]["preds"] = {}
        grouped[id]["preds"][gt_rel] = pred_rel
        if "logits" not in grouped[id]:
            grouped[id]["logits"] = {}
        grouped[id]["logits"][gt_rel] = logits
        if "data_row" not in grouped[id]:
            grouped[id]["data_row"] = row

    for id, gr_dict in grouped.items():
        ex = gr_dict["preds"]
        print("-"*100)
        gt_label = list(ex.keys())[0].split("__")[0]
        yes_labels = [False for _ in range(len(mapping))]
        pred_rel_type = None
        for comp_rel_type, yn_pred in ex.items():
            rel_type, q_rel_type = comp_rel_type.split("__")
            answ = "NO"
            if yn_pred == 1:
                answ = "YES"
            elif yn_pred == 2:
                answ = "OTHER"
            print(f"GT: {rel_type} | Q: {q_rel_type} | ANSW: {'YES' if yn_pred == 1 else 'NO'}")
            # YES has id 1
            if yn_pred == 1:
                yes_labels[mapping[q_rel_type]] = True
                pred_rel_type = q_rel_type
            elif yn_pred == 2:
                yes_labels[mapping["OTHER"]] = True
                pred_rel_type = "OTHER"

        out_gt_rels.append(mapping[gt_label])
        print(yes_labels)
        if sum(yes_labels) == 1:
            out_pred_rels.append(mapping[pred_rel_type])
            print(f"PRED: {pred_rel_type}")
        else:
            out_pred_rels.append(mapping["OTHER"])
            print(f"PRED: OTHER")
        # one per class (except OTHER)
        if "VAGUE" in mapping:
            # ignore OTHER and VAGUE
            assert len(ex) == len(mapping) - 2
        else:
            # ignore OTHER
            assert len(ex) == len(mapping) - 1

    return out_gt_rels, out_pred_rels, grouped

def get_output(pred_temp_rel: List[int], grouped_preds: Dict[Tuple[str, str], Dict[str, Any]], rev_mapping: Dict[int, str]) -> List[DataRow]:
    output = []
    for pred, (gr_pred_id, gr_pred) in zip(pred_temp_rel, grouped_preds.items()):
        row = gr_pred["data_row"]
        gr_pred_questions = gr_pred["preds"]
        gr_logits_questions = gr_pred["logits"]
        id = (f"{row.context.d_id}_{row.event1.sent_id}_{row.event1.offset}_{row.event1.e_id}", f"{row.context.d_id}_{row.event2.sent_id}_{row.event2.offset}_{row.event2.e_id}")
        assert id == gr_pred_id
        answers = {}
        for id, yn_pred in gr_pred_questions.items():
            _, q_rel_type = id.split("__")
            yn_str = "NO"
            if yn_pred == 1:
                yn_str = "YES"
            if yn_pred == 2:
                yn_str = "OTHER"
            answers[q_rel_type] = yn_str
        logits = {}
        for id, lg in gr_logits_questions.items():
            _, q_rel_type = id.split("__")
            logits[q_rel_type] = lg
        row.prediction = {
            "class": rev_mapping[pred],
            "answers": answers,
            "logits": logits
        }
        output.append(row)
    return output

def eval_train_loop(data, model):
    model.eval()
    loss_array = []
    with torch.no_grad():
        for sample in tqdm(data, unit="batch"):
            input_ids = sample["input_ids"].to(model.device)
            attention_mask = sample["attention_mask"].to(model.device)
            labels = input_ids.clone()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss

            loss_array.append(loss.item())

    return np.asarray(loss_array).mean()

def eval_loop(data, model, mapping, rel_type_order, dataset):
    model.eval()
    loss_array = []
    gt_temp_rel = []
    pred_temp_rel = []
    pred_logits = []
    ev_ids = []
    output = []
    rev_mapping = {v :k for k,v in mapping.items()}
    tmp_sample = []
    out_str_list = []
    n_labels = len(mapping.keys()) - 1 # ignore OTHER
    if dataset == "TB-DENSE":
        n_labels = len(mapping.keys()) - 2 # ignore OTHER, skip VAGUE (no question)

    with torch.no_grad():
        for sample in tqdm(data, desc="Evaluating", unit="batch"):
            questions = None
            if "questions_ids" in sample:
                # questions_sequential
                questions=sample["questions_ids"]

            loss, logits, outputs_str, out_logits = model(
                sample['input_ids'],
                sample['attention_mask'],
                sample['labels'],
                sample['masks'],
                sample['complementary_mask'],
                sample['to_duplicate'],
                questions=questions
            )

            out_str_list.extend(outputs_str)

            tmp_pred_rel = np.argmax(logits.cpu().numpy(), axis=1)
            pred_temp_rel.extend(tmp_pred_rel)
            pred_logits.extend(out_logits)
            if isinstance(model, Llama_Temp) and "questions" in model.prompt_type:
                if model.prompt_type == "questions_sequential":
                    ev_ids.extend([
                        (f"{row.context.d_id}_{row.event1.sent_id}_{row.event1.offset}_{row.event1.e_id}",
                         f"{row.context.d_id}_{row.event2.sent_id}_{row.event2.offset}_{row.event2.e_id}")
                        for row in sample["data_rows"]
                    ]*n_labels) # *n_labels -> one example will return n_labels labels (one for each question)
                    comp_rel_types = []
                    for q_rel_type in rel_type_order:
                        # assume batch_size == 1
                        rel_type = sample["data_rows"][0].label
                        comp_rel_types.append(f"{rel_type}__{q_rel_type}")
                    print(comp_rel_types)
                    gt_temp_rel.extend(comp_rel_types)
                    # keep data_rows
                    tmp_sample.extend([x for x in sample['data_rows']]*n_labels) # *n_labels -> one example will return n_labels labels (one for each question)
                else:
                    ev_ids.extend([
                        (f"{row.context.d_id}_{row.event1.sent_id}_{row.event1.offset}_{row.event1.e_id}",
                         f"{row.context.d_id}_{row.event2.sent_id}_{row.event2.offset}_{row.event2.e_id}")
                        for row in sample["data_rows"]
                    ])
                    gt_temp_rel.extend([row.label for row in sample["data_rows"]])
                    # keep data_rows
                    tmp_sample.extend([x for x in sample['data_rows']])
            else:
                gt_temp_rel.extend(sample['labels'].cpu().numpy())
                loss_array.append(loss.item())
                tmp_sample = [x for x in sample['data_rows']]

                # print("v"*100)
                # print(tmp_pred_rel)
                # print(out_logits)
                for id_elm, (elm, logits) in enumerate(zip(tmp_pred_rel, out_logits)):
                    # tmp_sample[id_elm].prediction = rev_mapping[elm]
                    tmp_sample[id_elm].prediction = {
                        "class": rev_mapping[elm],
                        "logits": logits
                    }
                    output.append(tmp_sample[id_elm])

    if isinstance(model, Llama_Temp) and "questions" in model.prompt_type:
        assert len(gt_temp_rel) == len(pred_temp_rel) == len(tmp_sample) == len(ev_ids) == len(pred_logits)
        gt_temp_rel, pred_temp_rel, grouped_preds = determine_preds(gt_temp_rel, pred_temp_rel, pred_logits, ev_ids, tmp_sample, mapping)
        # output with YES/NO
        output = get_output(pred_temp_rel, grouped_preds, rev_mapping)

    classes = list(mapping.keys())
    print(gt_temp_rel)
    print("pred before:", pred_temp_rel)
    if dataset == "TB-DENSE":
        # convert OTHER to VAGUE
        tmp = []
        for pred in pred_temp_rel:
            conv_pred = pred
            if rev_mapping[pred] == "OTHER":
                conv_pred = mapping["VAGUE"]
            tmp.append(conv_pred)
        pred_temp_rel = tmp
        # remove OTHER from classes
        assert classes[-1] == "OTHER"
        classes = classes[:-1]

    print("pred after:", pred_temp_rel)
    print(classes)
    print(list(range(len(classes))))
    print(classification_report(gt_temp_rel, pred_temp_rel, target_names=classes, labels=list(range(len(classes))), output_dict=False, zero_division=0))

    return classification_report(gt_temp_rel, pred_temp_rel, target_names=classes, labels=list(range(len(classes))), output_dict=True, zero_division=0), loss_array, output, out_str_list
    # return classification_report(gt_temp_rel, pred_temp_rel, target_names=list(mapping.keys()), labels=list(range(len(list(mapping.keys())))), output_dict=True, zero_division=0), loss_array, output, out_str_list


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
