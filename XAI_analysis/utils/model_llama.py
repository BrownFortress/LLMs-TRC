from pprint import pprint
from typing import Dict, List, Union
import torch
import torch.nn.functional as F, torch.nn as nn
import copy
import re
import numpy as np
from utils.chatgpt_model import ChatGPTModel

from utils.dataloader import B_INST, E_INST

class Llama_Temp(nn.Module):
    def __init__(self, pretrained_model, label_mapping, tokenizer, classifier=None, text_cls=False, prompting=True, max_new_tokens=10, prompt_type="class", sys_prompt_type=None, rel_type_order: List[str] = None, save_logits:bool = False):
        super(Llama_Temp, self).__init__()
        if prompting:
            self.model = pretrained_model
        else:
            self.model = copy.deepcopy(pretrained_model)
        self.text_classification = text_cls
        self.prompting = prompting
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.max_new_tokens = max_new_tokens
        self.save_logits = save_logits
        self.prompt_type = prompt_type
        self.sys_prompt_type = sys_prompt_type
        self.rel_type_order = rel_type_order
        self.softmax = nn.Softmax(dim=0)
        self.q_label_mapping = {
            "NO": 0,
            "YES": 1,
            "OTHER": 2,
        }

    def _get_label_class(self, outputs_str) -> List[int]:
        labels = []
        for output in outputs_str:
            out_lab = "OTHER"
            for lab in self.label_mapping.keys():
                if lab in output:
                    out_lab = lab
                    break
            labels.append(self.label_mapping[out_lab])
        return labels

    def _get_label_questions(self, outputs_str, map=True) -> List[Union[int,str]]:
        labels = []
        for output in outputs_str:
            out_lab = "OTHER"
            if re.search(r"\byes\b", output.lower()) is not None:
                out_lab = "YES"
            elif re.search(r"\bno\b", output.lower()) is not None:
                out_lab = "NO"
            print(out_lab)
            if map:
                out_lab = self.q_label_mapping[out_lab]
            labels.append(out_lab)
        return labels

    def _fwd(self, input_ids, attention_mask):
        # only get generated tokens
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        out_pred = outputs.sequences[:,input_ids.shape[1]:]

        if not isinstance(self.model, ChatGPTModel):
            out_pred = outputs.sequences[:,input_ids.shape[1]:]
            # scores only for generated tokens
            out_logits = [None]
            if self.save_logits:
                out_logits = [outputs.scores]
            outputs_str = self.tokenizer.batch_decode(out_pred)
            print(self.tokenizer.batch_decode(input_ids)[0])
        else:
            out_logits = [None]
            if self.save_logits:
                out_logits = [tuple([None for _ in range(self.max_new_tokens)])]
            outputs_str = outputs

        print(outputs_str)
        print("-"*100)

        if self.prompt_type == "class":
            labels = self._get_label_class(outputs_str)
            logits = torch.zeros(len(labels), len(self.label_mapping.keys()))
        elif "questions" in self.prompt_type:
            labels = self._get_label_questions(outputs_str)
            logits = torch.zeros(len(labels), len(self.q_label_mapping.keys()))
        return labels, logits, outputs_str, out_logits
    
    def _fwdIG(self, input_ids, attention_mask):
        # only get generated tokens            
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        prob = self.softmax(outputs.logits[0][-1])
        
        return prob
    
    def _fwd_questions_seq(self, input_ids, attention_mask, questions):
        is_api_model = isinstance(self.model, ChatGPTModel)
        outputs_str_list = []
        outputs_logits_list = []
        # assume batch size 1
        assert len(questions) == 1
        questions = questions[0]
        q_a_embs = [] # will contain q1, a1, q2, a2, ...
        labels = []
        # input_ids + q1
        # input_ids + q1 + a1 + q2
        # input_ids + q1 + a1 + q2 + a2 + q3
        eos_bos_emb  = torch.LongTensor([self.tokenizer.eos_token_id, self.tokenizer.bos_token_id])
        b_inst_emb = self.tokenizer(B_INST, padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        e_inst_emb = self.tokenizer(E_INST + " ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        for i, q_rel_type in enumerate(self.rel_type_order):
            question_emb = questions[q_rel_type]
            if self.sys_prompt_type is not None:
                if self.sys_prompt_type == "single_inst":
                    q_a_embs.append(question_emb.unsqueeze(0))
                    q_a_embs.append(e_inst_emb.unsqueeze(0))
                elif self.sys_prompt_type == "question_inst":
                    if i == 0:
                        question_emb = torch.cat([question_emb.to(self.model.device), e_inst_emb.to(self.model.device)], dim=0).type(torch.LongTensor)
                    else:
                        question_emb = torch.cat([eos_bos_emb.to(self.model.device), b_inst_emb.to(self.model.device), question_emb.to(self.model.device), e_inst_emb.to(self.model.device)], dim=0).type(torch.LongTensor)
                    q_a_embs.append(question_emb.unsqueeze(0))
            else:
                q_a_embs.append(question_emb.unsqueeze(0))

            # print("input ids:", input_ids.device)
            # print("q_a_embs:", [e.device for e in q_a_embs])
            # input_ids_with_question = torch.cat([input_ids] + q_a_embs, dim=1).type(torch.LongTensor)
            input_ids_with_question = torch.cat([input_ids.to(self.model.device)] + [e.to(self.model.device) for e in q_a_embs], dim=1).type(torch.LongTensor)
            attention_mask_with_question = torch.ones(input_ids_with_question.shape)
            attention_mask_with_question[:,:input_ids.shape[1]] = attention_mask
            outputs = self.model.generate(
                input_ids_with_question.to(self.model.device),
                attention_mask=attention_mask_with_question.to(self.model.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            # print(outputs.score)
            # print( self.tokenizer.batch_decode(out_pred))
            # a = 0/0
            # print("-"*100)
            if not is_api_model:
                out_pred = outputs.sequences[:,input_ids_with_question.shape[1]:]
                outputs_str = self.tokenizer.batch_decode(out_pred)
                # scores only for generated tokens
                out_logits = [None]
                if self.save_logits:
                    out_logits = [outputs.scores]
            else:
                out_logits = [None]
                if self.save_logits:
                    out_logits = [tuple([None for _ in range(self.max_new_tokens)])]
                outputs_str = outputs
            outputs_str_list.extend(outputs_str)
            outputs_logits_list.extend(out_logits)
            # print(self.tokenizer.batch_decode(input_ids_with_question)[0])
            # print(outputs_str)
            answ_str = self._get_label_questions(outputs_str, map=False)[0]
            answ_emb = self.tokenizer(f" {answ_str} ", padding=False, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            if self.sys_prompt_type is not None:
                if self.sys_prompt_type == "single_inst":
                    # remove [/INST]
                    q_a_embs.pop()

            q_a_embs.append(answ_emb.unsqueeze(0))
            labels.append(self.q_label_mapping[answ_str])
        logits = torch.zeros(len(labels), len(self.q_label_mapping.keys()))

        return labels, logits, outputs_str_list, outputs_logits_list

    def _fwd_questions_seq_text(self, input_ids, attention_mask, questions):
        outputs_str_list = []
        outputs_logits_list = []
        # assume batch size 1
        assert len(questions) == 1
        questions = questions[0]
        q_a_embs = [] # will contain q1, a1, q2, a2, ...
        labels = []
        # input_ids + q1
        # input_ids + q1 + a1 + q2
        # input_ids + q1 + a1 + q2 + a2 + q3
        for q_rel_type in self.rel_type_order:
            question_emb = questions[q_rel_type]
            q_a_embs.append(question_emb)

            print("-"*100)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                q_a_list=q_a_embs
            )
            out_logits = [tuple([None for _ in range(self.max_new_tokens)])]
            outputs_str = outputs
            outputs_str_list.extend(outputs_str)
            outputs_logits_list.extend(out_logits)
            # pprint(input_ids)
            # print(outputs_str)
            answ_str = self._get_label_questions(outputs_str, map=False)[0]

            q_a_embs.append(answ_str)
            labels.append(self.q_label_mapping[answ_str])
        logits = torch.zeros(len(labels), len(self.q_label_mapping.keys()))

        return labels, logits, outputs_str_list, outputs_logits_list

    def forward(self, input_ids, attention_mask, labels, span_mask, complementary_mask, to_duplicate, questions=None, IG=False, unroll=False , device="cuda", **args):
        if self.prompting:
            out_logits = None
            if IG:
                return self._fwdIG(input_ids, attention_mask)
            if unroll:
                
                if questions is None:
                    out_ids = []
                    tmp_inputs_ids = input_ids
                    tmp_attention_mask = attention_mask
                    
                    for x in range(self.max_new_tokens):
                        pred = self._fwdIG(tmp_inputs_ids, tmp_attention_mask)
                        token = pred.cpu().argmax(dim=0)
                        print("Id token", token)
                        # tmp_inputs_ids = torch.cat([tmp_inputs_ids, torch.tensor([[token]])], dim=1)
                        # tmp_attention_mask = torch.cat([tmp_attention_mask,  torch.tensor([[1]])], dim=1)
                        tmp_inputs_ids = torch.cat([tmp_inputs_ids, torch.tensor([[token]]).to(device)], dim=1)
                        tmp_attention_mask = torch.cat([tmp_attention_mask,  torch.tensor([[1]]).to(device)], dim=1)
                        out_ids.append(token)
                        lab_stringfy = self.tokenizer.decode(out_ids).strip()
                        for lab in self.label_mapping.keys():
                            if lab == lab_stringfy:
                                return x, lab_stringfy
          
                    return None, lab_stringfy
                else:   
                    labels, logits, out_str, out_logits = self._fwd_questions_seq(input_ids, attention_mask, questions)
                    
                    print(out_str)
                    print(np.argmax(labels))
                    a = 0/0
                    return None, out_str
            if questions is None:
                labels, logits, out_str, out_logits = self._fwd(input_ids, attention_mask)
            else:
                print("type input ids: ", type(input_ids[0]))
                if not isinstance(input_ids[0], dict):
                    labels, logits, out_str, out_logits = self._fwd_questions_seq(input_ids, attention_mask, questions)
                else:
                    labels, logits, out_str, out_logits = self._fwd_questions_seq_text(input_ids, attention_mask, questions)
            for i, lab in enumerate(labels):
                logits[i,lab] = 1
            logits_temp = logits.cpu()
            loss = torch.tensor(0)
        else:
            pass
        return loss, logits_temp, out_str, out_logits
