import math

import torch
import torch.nn.functional as F, torch.nn as nn
import copy
from pprint import pprint

class FeedForward(nn.Module):
    def __init__(self, layers, out_labels):
        super(FeedForward, self).__init__()

        activation_functions ={'relu': nn.ReLU()}
        modules = []
        n_layers = len(layers)
        for i, layer in enumerate(layers):
            if layer['type'] == 'linear':
                if i == n_layers -1:
                    modules.append(nn.Linear(layer['in'], out_labels))
                else:
                    modules.append(nn.Linear(layer['in'], layer['out']))
            elif layer['type'] == 'dropout':
                modules.append(nn.Dropout(layer['prob']))
            else:
                modules.append(activation_functions[layer['type'].lower()])

        self.ff = nn.Sequential(*modules)

    def forward(self, x):
        return self.ff(x)

class LoRaModel(nn.Module):
    def __init__(self, pretrained_model_lora_model, classifier):
        super(LoRaModel, self).__init__()
        self.model = pretrained_model_lora_model
        # self.dropout = nn.Dropout(0.3)
        self.loss_temp = nn.CrossEntropyLoss()
        self.classifier= classifier

    def forward(self, input_ids, attention_mask, labels, span_mask,  complementary_mask, to_duplicate, **args):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,  **args)
        lhs = outputs.hidden_states[-1] # This should be the last layer
        
        span_mask_unsqueezed = span_mask.unsqueeze(2)
        comp_mask_unsqueezed = complementary_mask.unsqueeze(2)
        duplicated_lhs = torch.repeat_interleave(lhs, to_duplicate, dim=0)
        tmp = duplicated_lhs * span_mask_unsqueezed + comp_mask_unsqueezed # We extract the tokens that we want to merge
        pooled = tmp.max(dim=1)[0]
        reshape_pooled = pooled.reshape(int(duplicated_lhs.size()[0]/2), 2, duplicated_lhs.size()[-1]) # The dim=1 we have the representation of events
        new_pooled = torch.cat([reshape_pooled[:, 0], reshape_pooled[:,1]], dim=1)
        
        logits_temp = self.classifier(new_pooled)
        
        loss = self.loss_temp(logits_temp.cpu(), labels)
        return loss, logits_temp
    
class WrapperClassifier(nn.Module):
    def __init__(self, classifier):
        super(WrapperClassifier, self).__init__()

        self.loss_temp = nn.CrossEntropyLoss()
        self.classifier= classifier

    def forward(self, embeddings, attention_mask, labels, span_mask,  complementary_mask, to_duplicate, **args):

        # Embeddings are the max pooled contatenated embeddigns coming from EncoderFrozen
        logits_temp = self.classifier(embeddings)
        
        loss = self.loss_temp(logits_temp.cpu(), labels)
        return loss, logits_temp
    
class EncoderFrozen(nn.Module):
    def __init__(self, pretrained_model):
        super(EncoderFrozen, self).__init__()
        self.model = pretrained_model

    def forward(self, input_ids, attention_mask,  span_mask,  complementary_mask, to_duplicate, **args):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,  **args)
        lhs = outputs.hidden_states[-1] # This should be the last layer
        span_mask_unsqueezed = span_mask.unsqueeze(2)
        comp_mask_unsqueezed = complementary_mask.unsqueeze(2)
        duplicated_lhs = torch.repeat_interleave(lhs, to_duplicate, dim=0)
        tmp = duplicated_lhs * span_mask_unsqueezed + comp_mask_unsqueezed # We extract the tokens that we want to merge
        pooled = tmp.max(dim=1)[0]
        reshape_pooled = pooled.reshape(int(duplicated_lhs.size()[0]/2), 2, duplicated_lhs.size()[-1]) # The dim=1 we have the representation of events
        new_pooled = torch.cat([reshape_pooled[:, 0], reshape_pooled[:,1]], dim=1)

        return new_pooled.squeeze(0)
    
class EncoderForRAG(nn.Module):
    def __init__(self, pretrained_model):
        super(EncoderForRAG, self).__init__()
        self.model = pretrained_model

    def forward(self, input_ids, attention_mask, **args):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,  **args)
        lhs = outputs.hidden_states[-1] # This should be the last layer
        return lhs.squeeze(0)

class BERT_Temp(nn.Module):
    def __init__(self, pretrained_model, classifier, text_cls=False, cls_as_context=False):
        super(BERT_Temp, self).__init__()
        self.model = copy.deepcopy(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.loss_temp = nn.CrossEntropyLoss()
        self.text_classification = text_cls
        self.classifier= classifier
        self.cls_as_context = cls_as_context

    def forward(self, input_ids, attention_mask, labels, span_mask,  complementary_mask, to_duplicate, **args):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,  **args)
        cls = outputs.pooler_output
        lhs = outputs.last_hidden_state

        if not self.text_classification:
            span_mask_unsqueezed = span_mask.unsqueeze(2)
            comp_mask_unsqueezed = complementary_mask.unsqueeze(2)
            duplicated_lhs = torch.repeat_interleave(lhs, to_duplicate, dim=0)
            tmp = duplicated_lhs * span_mask_unsqueezed + comp_mask_unsqueezed # We extract the tokens that we want to merge
            pooled = tmp.max(dim=1)[0]
            reshape_pooled = pooled.reshape(int(duplicated_lhs.size()[0]/2), 2, duplicated_lhs.size()[-1]) # The dim=1 we have the representation of events
            if self.cls_as_context:
                new_pooled = torch.cat([reshape_pooled[:, 0], reshape_pooled[:,1], cls], dim=1)
            else:
                new_pooled = torch.cat([reshape_pooled[:, 0], reshape_pooled[:,1]], dim=1)
            logits_temp = self.classifier(new_pooled)
            logits_temp = self.classifier(new_pooled)

        else:
            logits_temp = self.classifier(cls)

        loss = self.loss_temp(logits_temp.cpu(), labels)
        return loss, logits_temp

class BERT_Temp_Analysis(nn.Module):
    def __init__(self, pretrained_model, classifier, text_cls=False, cls_as_context=False):
        super(BERT_Temp_Analysis, self).__init__()
        self.model = copy.deepcopy(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.loss_temp = nn.CrossEntropyLoss()
        self.text_classification = text_cls
        self.classifier= classifier
        self.cls_as_context = cls_as_context

    def forward(self, input_ids, attention_mask, span_mask,  complementary_mask, to_duplicate, **args):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,  **args)
        cls = outputs.pooler_output
        lhs = outputs.last_hidden_state

        if not self.text_classification:
            span_mask_unsqueezed = span_mask.unsqueeze(2)
            comp_mask_unsqueezed = complementary_mask.unsqueeze(2)
            duplicated_lhs = torch.repeat_interleave(lhs, to_duplicate, dim=0)
            tmp = duplicated_lhs * span_mask_unsqueezed + comp_mask_unsqueezed # We extract the tokens that we want to merge
            pooled = tmp.max(dim=1)[0]
            reshape_pooled = pooled.reshape(int(duplicated_lhs.size()[0]/2), 2, duplicated_lhs.size()[-1]) # The dim=1 we have the representation of events
            if self.cls_as_context:
                new_pooled = torch.cat([reshape_pooled[:, 0], reshape_pooled[:,1], cls], dim=1)
            else:
                new_pooled = torch.cat([reshape_pooled[:, 0], reshape_pooled[:,1]], dim=1)
            logits_temp = self.classifier(new_pooled)

        else:
            logits_temp = self.classifier(cls)

        return logits_temp # This is the only difference
