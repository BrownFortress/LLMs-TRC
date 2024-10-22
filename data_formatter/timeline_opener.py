import pandas as pd
import pickle
import os
from sandbox.data_structures import Event, Document, Relation
import re

BASE_PATH = 'original_data/TIMELINE'
ANNOTATION_PATH = os.path.join(BASE_PATH, 'annotations')
CORPUS_PATH = os.path.join(BASE_PATH, 'corpus')

def remove_punctuation(test_str):
    # Using filter() and lambda function to filter out punctuation characters
    result = "".join(
        filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), test_str)
    )
    return result


mapping = {"a": "AFTER", "b":"BEFORE", "e":"EQUAL", "v":"VAGUE"}
annotations = {}
for file in os.listdir(ANNOTATION_PATH):
    if 'Events' in file:
        annotations['events'] = pd.read_csv(os.path.join(ANNOTATION_PATH, file))
    elif 'Relations' in file:
        annotations['relations'] = pd.read_csv(os.path.join(ANNOTATION_PATH, file))
    elif 'boundaries' in file:
        annotations['boundaries'] = pd.read_csv(os.path.join(ANNOTATION_PATH, file))
raw_documents = {}
for file in os.listdir(CORPUS_PATH):
    with open(os.path.join(CORPUS_PATH, file), "r") as f:
        raw_documents[file.split(".")[0]] = f.read()

documents = {}
for d_id, raw_doc in sorted(raw_documents.items()):
    tokenized = raw_doc.split()
    sentences = []
    # print(d_id)
    bounds =  list(annotations['boundaries'][annotations['boundaries']['Document ID'] == d_id].values)
    # print(bounds)
    iter_bounds = iter(bounds)
    bound = next(iter_bounds)
    start = bound[2].strip()
    end = bound[3].strip()
    tmp_sentence = []
    # print(start, end)
    missing_tokens = []
    is_start = False
    for x in tokenized:
        if x == start:
            is_start = True
            # print("Start")
            # print(d_id, start, end)
        if is_start:
            tmp_sentence.append(x)

        if x == end and is_start:
            is_start = False
            assert len(tmp_sentence) != 0
            sentences.append(tmp_sentence)
            tmp_sentence = []
            if len(sentences) != len(bounds):
                bound = next(iter_bounds)
                start = bound[2].strip()
                end = bound[3].strip()
            else:
                break
            
    if len(sentences) != len(bounds):
        print(d_id, start, end)
    assert len(sentences) == len(bounds)
    documents[d_id] = Document(d_id, None, None, sentences, None, sum(sentences,[]), [], None, None)
    
events_data= {}

for event_row in annotations['events'].values:
    d_id = event_row[0]
    sent_id = int(event_row[1].replace("S","")) - 1
    e_id = event_row[2]
    token_event = event_row[3].strip()
    occurence = event_row[4]
    event_to_look_at = 0 if occurence == 0 else occurence-1
    doc = documents[d_id]
    sentence = [x for x in doc.text[sent_id]]
    if occurence >= 1:
        occurence += 1
    txt = " ".join(sentence)
    split_text = re.split(token_event, txt)
    # assert len(split_text) > 2
    events = []
    counter = -1
    condition = (d_id == "Document008" and event_row[1] == "S9" and False)
    for id_s, span in enumerate(split_text):
        split_span = span.strip().split()
        if len(split_span) > 0:
            if split_span[0] in [",", ".", "!", "?"]:
                split_span.pop(0)
        if condition:
            print(split_span, id_s, len(split_span))
            print(counter)
            # print(txt)
            # print(split_text, token_event)
        if id_s < len(split_text)-1:
            counter += len(split_span)
            start = counter
            if condition:
                print(counter)
            counter += len(token_event.split())
            if condition:
                print(counter)
            end = counter 
            events.append((start+1, end+1))
    if condition:
        print("="*89)
        print(token_event, events, sentence)
    
    if len(events) != 0:
        event_span = []
        if event_to_look_at >= len(events):
            print(events)
            print("There is no Event:",d_id, event_row[1], sentence, token_event)
        else:
            event_span = events[event_to_look_at]
        
        assert e_id not in events_data
        
        events_data[e_id] = Event(e_id, " ".join(sentence[event_span[0] : event_span[1]]), None, None, slice(event_span[0], event_span[1]), sent_id, sentence, None, None)

        documents[d_id].events.append(Event(e_id, token_event, None, None, slice(event_span[0], event_span[1]), sent_id, sentence, None, None))
        
        if remove_punctuation(token_event) != remove_punctuation(" ".join(sentence[documents[d_id].events[-1].offset]).strip()):
            print("Miss match",d_id, event_span,event_row[1], "GT:", token_event.strip(),  "P:", " ".join(sentence[event_span[0]: event_span[1]]).strip(), sentence)
    else:
         print("Event not found",d_id, event_span,event_row[1], "GT:", token_event.strip()," ".join(sentence))
    # assert token_event == remove_punctuation(" ".join(sentence[documents[d_id].events[-1].offset]))
    
for rel_row in annotations['relations'].values:
    d_id = rel_row[0]
    e1 = rel_row[1]
    e2 = rel_row[2]
    eid1 = rel_row[3]
    eid2 = rel_row[4]
    label =mapping[rel_row[5]]

    
    if documents[d_id].temporal_relations is  None:
        documents[d_id].temporal_relations = {}
    if label not in  documents[d_id].temporal_relations:
        documents[d_id].temporal_relations[label] = []
    documents[d_id].temporal_relations[label].append(Relation(label, events_data[eid1], events_data[eid2]))
    
with open("original_data/dev.pickle", 'rb') as f:
    dev = pickle.load(f)
with open("original_data/test.pickle", 'rb') as f:
    test = pickle.load(f)
with open("original_data/train.pickle", 'rb') as f:
    train = pickle.load(f)
    
splits = {}
splits['train'] = set([x[0] for x in train])
splits['valid'] = set([x[0] for x in dev])
splits['test'] = set([x[0] for x in test])

output_files = {"train":[], "valid":[], "test":[]}
for k, v in documents.items():
    if k in splits["train"]:
        output_files["train"].append(v)
    elif k in splits["valid"]:
        output_files["valid"].append(v)
    elif k in splits["test"]:
        output_files["test"].append(v)
    else:
        print("ERROR:", k)

for k, v in output_files.items():
    with open("data/TIMELINE/" + k + ".pkl", "wb") as f:
        pickle.dump(v, f, pickle.HIGHEST_PROTOCOL)