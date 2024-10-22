from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import classification_report


def train_loop(data, optimizer, optimizer_cls, scheduler, model):
    model.train()
    loss_array = []
    for sample in tqdm(data, desc="Training", unit="batch"):
        loss, _ = model(sample['input_ids'], sample['attention_mask'], sample['labels'], sample['masks'], sample['complementary_mask'], sample['to_duplicate'])
        loss_array.append(loss.item())
        optimizer.zero_grad()  # Zeroing the gradient
        if optimizer_cls != None:
           optimizer_cls.zero_grad()

        loss.backward()  # Compute the gradient, deleting the computational graph
        optimizer.step()
        if optimizer_cls != None:
            optimizer_cls.step()
        if scheduler != None:
           scheduler.step()
           
    return loss_array


def eval_loop(data, model, mapping):
    model.eval()
    loss_array = []
    gt_temp_rel = []
    pred_temp_rel = []
    output = []
    rev_mapping = {v :k for k,v in mapping.items()}
    with torch.no_grad():
        for sample in tqdm(data, desc="Evaluating", unit="batch"):
         
            loss, logits = model(sample['input_ids'], sample['attention_mask'], sample['labels'], sample['masks'], sample['complementary_mask'], sample['to_duplicate'])
            tmp_pred_rel = np.argmax(logits.cpu().numpy(), axis=1)

            gt_temp_rel.extend(sample['labels'].cpu().numpy())
            pred_temp_rel.extend(tmp_pred_rel)
            loss_array.append(loss.item())
            tmp_sample = [x for x in sample['data_rows']]

            for id_elm, elm in enumerate(tmp_pred_rel):
                tmp_sample[id_elm].prediction = rev_mapping[elm]
                output.append(tmp_sample[id_elm])
    print(len(gt_temp_rel))
    return classification_report(gt_temp_rel, pred_temp_rel, target_names=list(mapping.keys()), labels=list(range(len(list(mapping.keys())))), output_dict=True, zero_division=0), loss_array, output



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
