import numpy as np
import torch 

def add_token(tok, token, model, llama=False):
    tok.add_tokens([token]) # use as zero-baseline for I.G.
    model.model.resize_token_embeddings(len(tok)) 
    with torch.no_grad():
        if llama:
            model.model.model.embed_tokens.weight[-1, :] = torch.zeros([model.model.config.hidden_size])
        else:  
            model.model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.model.config.hidden_size])
    

def aggregate(attributions, input_ids, tok, aggregator='mean', untokenize=False, splitter="Ġ"):
    new_attributions = []
    new_tokens = []
    special_tokens = [tok.bos_token_id, tok.eos_token_id]
    for id_ex, sequence in enumerate(input_ids):
        tokens = []
        tmp_attributions = []
        subtokens = tok.convert_ids_to_tokens(sequence)
        for id_token, subtoken in enumerate(subtokens):
            if sequence[id_token] != tok.pad_token_id:
                attribution = attributions[id_ex][id_token].sum()
                if untokenize:
                    if splitter in subtoken  or sequence[id_token] in special_tokens:
                        tokens.append(subtoken)
                        tmp_attributions.append([attribution.item()])
                    else:
                        if sequence[id_token-1] in special_tokens:
                            tokens.append(subtoken)
                            tmp_attributions.append([attribution.item()])
                        else:
                            tokens[-1] += subtoken
                            tmp_attributions[-1].append(attribution.item())
                else:
                    tokens.append(subtoken)
                    tmp_attributions.append(attribution.item())
        new_tokens.append(tokens)
        if untokenize:
            if aggregator == 'mean':
                tmp_attributions = [np.asanyarray(x).mean() for x in tmp_attributions]
            elif aggregator == 'max': 
                tmp_attributions = [np.asanyarray(x).max() for x in tmp_attributions]
            elif aggregator == 'sum': 
                tmp_attributions = [np.asanyarray(x).sum() for x in tmp_attributions]
            else:
                raise Exception(f"Sorry, {aggregator} not implemented yet.")

        new_attributions.append(tmp_attributions)
    return new_attributions, new_tokens

def aggregateLLAMA(attributions, input_ids, tok, aggregator='mean', untokenize=False, splitter="_"):
    new_attributions = []
    new_tokens = []
    special_tokens = [tok.bos_token_id, tok.eos_token_id]
    for id_ex, sequence in enumerate(input_ids):
        tokens = []
        tmp_attributions = []
        subtokens = sequence
        for id_token, subtoken in enumerate(subtokens):
            if sequence[id_token] != tok.pad_token_id:
                attribution = attributions[id_ex]
                if untokenize:
                    if splitter in subtoken and len(subtoken) > 1:
                        tokens[-1] += subtoken
                        tmp_attributions[-1].append(attribution)
                    else:
                        if sequence[id_token-1] in special_tokens:
                            tokens.append(subtoken)
                            tmp_attributions.append([attribution])
                        else:
                            tmp_attributions.append([attribution])
                else:
                    tokens.append(subtoken)
                    tmp_attributions.append(attribution.item())
        new_tokens.append(tokens)
        if untokenize:
            if aggregator == 'mean':
                tmp_attributions = [np.asanyarray(x).mean() for x in tmp_attributions]
            elif aggregator == 'max': 
                tmp_attributions = [np.asanyarray(x).max() for x in tmp_attributions]
            elif aggregator == 'sum': 
                tmp_attributions = [np.asanyarray(x).sum() for x in tmp_attributions]
            else:
                raise Exception(f"Sorry, {aggregator} not implemented yet.")

        new_attributions.append(tmp_attributions)
    return new_attributions, new_tokens

def construct_references(input_ids, tok):
    refs = []
    max_len = max([len(x) for x in input_ids])
    assert tok.convert_tokens_to_ids("[ABLATE_WORD]") != tok.unk_token_id
    
    null_token_id = tok.convert_tokens_to_ids("[ABLATE_WORD]")
    for s in input_ids:
        diff = max_len - len(s)
        refs.append([tok.cls_token_id] + [null_token_id] * (len(s)-2)  + [tok.sep_token_id] + [null_token_id] * diff)
        # refs.append([cls_token_id] + [null_token_id] * (len(s) + [sep_token_id] + [null_token_id] * (model_max_length - 2 - len(s)))
    return refs
