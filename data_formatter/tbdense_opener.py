from argparse import ArgumentParser
from collections import Counter
import xml.etree.ElementTree as ET
import os
from bs4 import BeautifulSoup
import pandas as pd
from pprint import pprint
import nltk
import spacy
import json
from tqdm import tqdm
import pickle
from sandbox.data_structures import Event, Document, Relation, TemporalFunction


def remove_punctuation(test_str):
    # Using filter() and lambda function to filter out punctuation characters
    result = "".join(
        filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), test_str)
    )
    return result


def get_pos_from_contents(elem, contents):
    index = 0
    for x in contents:
        if str(x) == str(elem):
            return index
        index += 1
    print("Value {elem} not found")
    return None


def reformat_sentence(to_tokenize, events_in_sentence, all_events):
    new_sent = []
    for pos_id, chunck in enumerate(to_tokenize):
        if pos_id in events_in_sentence:
            for ev in events_in_sentence[pos_id]:
                start_pos = len(new_sent)
                new_sent.extend(
                    chunck.split()
                )  # There are 4 events in the test set that are more that 1 tokens
                end_pos = len(new_sent)
                all_events[ev].offset = slice(start_pos, end_pos)
                try:
                    assert (
                        " ".join(new_sent[all_events[ev].offset])
                        == all_events[ev].token
                    )
                except:
                    print("SENTENCE ALERT!")
                    print(to_tokenize)
                    print(all_events[ev])
                    print(new_sent)
        else:
            tokenized_chunk = chunck.split()
            new_sent.extend(tokenized_chunk)

    assert new_sent[-1] == to_tokenize[-1].split()[-1]
    return new_sent


def reformat_paragraph(to_tokenize, events_in_sentence, all_events):
    new_para = []
    for pos_id, chunck in enumerate(to_tokenize):
        if pos_id in events_in_sentence:
            for ev in events_in_sentence[pos_id]:
                start_pos = len(new_para)

                new_para.extend(chunck.split())
                end_pos = len(new_para)

                all_events[ev].offset_paragraph = slice(start_pos, end_pos)
                try:
                    assert (
                        " ".join(new_para[all_events[ev].offset_paragraph])
                        == all_events[ev].token
                    )
                except:
                    print("PARAGRAPH ALERT!")
                    print(to_tokenize)
                    print(all_events[ev])
                    print(new_para)
        else:
            tokenized_chunk = chunck.split()
            new_para.extend(tokenized_chunk)

    assert new_para[-1] == to_tokenize[-1].split()[-1]
    return new_para


parser = ArgumentParser(description="Prepare TB-DENSE data")
parser.add_argument("--max_docs", type=int, help="Maximum number of documents to prepare for each split.")
args = parser.parse_args()
max_docs = float("inf")
if args.max_docs is not None:
    max_docs = args.max_docs

dataset = {}
BASE_PATH = "original_data/TempEval"
partitions = ["TimeBank", "AQUAINT", "te3-platinum"]
dataset = {}
for partition in partitions:
    for file in os.listdir(os.path.join(BASE_PATH, partition)):
        if "tml" in file:
            new_file = file.replace(".tml", "")
            dataset[new_file] = BeautifulSoup(
                open(os.path.join(BASE_PATH, partition, file)).read(), "xml"
            )

matres_files = ["aquaint.txt", "platinum.txt", "timebank.txt"]

matres = {}
BASE_MATRES_PATH = "original_data/TB-DENSE"

splits = {"train": [], "valid": [], "test": []}
split_ids = json.loads(open(os.path.join(BASE_MATRES_PATH, 'TB-DENSE_splits.json'), 'r').read())

df = pd.read_csv(os.path.join(BASE_MATRES_PATH, 'TimebankDense_annotation.txt'), sep="\t", header=None)
mapping = {'v': "VAGUE", 'a':'AFTER', 'b':"BEFORE", 'ii':'IS_INCLUDED', 'i':'INCLUDES', 's':'SIMULTANEOUS'}

for row in df.iterrows():
    doc_id = row[1][0]
    eiid1 = row[1][1]
    eiid2 = row[1][2]
    ty = mapping[row[1][3]]
    if doc_id in split_ids['train']:
        splits["train"].append(doc_id)
    elif doc_id in split_ids['dev']:
        splits["valid"].append(doc_id)
    elif doc_id in split_ids['test']:
        splits["test"].append(doc_id)
    else:
        print(doc_id)
    if doc_id not in matres:
        matres[doc_id] = {}
    if ty not in matres[doc_id]:
        matres[doc_id][ty] = []
    matres[doc_id][ty].append(
        {"eid1": eiid1, "eid2": eiid2}
    )

for file in os.listdir(BASE_MATRES_PATH):
    if file in matres_files:
        df = pd.read_csv(os.path.join(BASE_MATRES_PATH, file), sep="\t", header=None)
        

documents = {}
differences_paragraph = []
differences_sentences = []
n_docs = {
    split: 0 for split in splits.keys()
}
for d_id, doc in tqdm(dataset.items()):
    title = doc.find("TITLE").get_text() if doc.find("TITLE") is not None else None
    if doc.find("EXTRAINFO") is not None:
        extra_info = doc.find("EXTRAINFO").get_text()
    else:
        extra_info = ""

    plain_text = doc.find("TEXT").get_text()
    turns = [x for x in plain_text.split("\n") if x != ""]
    events_in_text = doc.find("TEXT").find_all("EVENT")

    full_tokenized_text = []
    for t in turns:
        full_tokenized_text.extend(remove_punctuation(t).split())

    timeExpression_in_text = doc.find("TEXT").find_all("TIMEX3")
    signals_in_text = doc.find("TEXT").find_all("SIGNAL")

    event_instances = doc.find_all("MAKEINSTANCE")
    contents = doc.find("TEXT").contents

    mapping = {}
    for instance in event_instances:
        eventID = instance.get("eventID")
        tense = instance.get("tense")
        pos = instance.get("pos")
        mapping[eventID] = {"eid": eventID, "tense": tense, "pos": pos}

    new_content = []
    paragraphs = []
    tmp_content = []
    tmp_paragraph = []
    added_events = {}
    added_events_paragraph = {}
    events = {}

    n_paragraphs = len(turns)
    n_events = 0

    for id_content, content in enumerate(contents):
        if content in events_in_text:
            n_events += 1
            event = events_in_text[events_in_text.index(content)]
            if event.get("eid") in mapping:
                ev = mapping[event.get("eid")]
                token = event.get_text().strip()
                e_class = event.get("class")
                tmp_content.append(token)
                tmp_paragraph.append(token)
                pos_in_sentence = len(tmp_content) - 1
                pos_in_paragraph = len(tmp_paragraph) - 1

                sentence_id = len(new_content)
                paragraph_id = len(paragraphs)

                if pos_in_sentence not in added_events:
                    added_events[pos_in_sentence] = []
                if pos_in_paragraph not in added_events_paragraph:
                    added_events_paragraph[pos_in_paragraph] = []

                added_events[pos_in_sentence].append(ev["eid"])
                added_events_paragraph[pos_in_paragraph].append(ev["eid"])

                events[ev["eid"]] = Event(
                    ev["eid"],
                    token,
                    ev["tense"],
                    ev["pos"],
                    pos_in_sentence,
                    sentence_id,
                    [],
                    pos_in_paragraph,
                    paragraph_id,
                )
            else:
                tmp_content.append(token)
                tmp_paragraph.append(token)

        elif content in timeExpression_in_text:
            id_t = content.get("tid")
            ty = content.get("type")
            val = content.get("value")
            token_time = content.get_text()
            tmp_content.append(token_time)
            tmp_paragraph.append(token_time)
            pos_in_sentence = len(tmp_content) - 1
            pos_in_paragraph = len(tmp_paragraph) - 1
            sentence_id = len(new_content)
            paragraph_id = len(paragraphs)

            if pos_in_sentence not in added_events:
                added_events[pos_in_sentence] = []
            if pos_in_paragraph not in added_events_paragraph:
                added_events_paragraph[pos_in_paragraph] = []

            added_events[pos_in_sentence].append(id_t)
            added_events_paragraph[pos_in_paragraph].append(id_t)
            # In TB-DENSE events and time expressions are on the same level
            events[id_t] = Event( 
                id_t,
                token_time,
                ty,
                val,
                pos_in_sentence,
                sentence_id,
                [],
                offset_paragraph=pos_in_paragraph,
                paragraph_id=paragraph_id,
            )

        elif content in signals_in_text:
            tmp_content.append(content.get_text())
            tmp_paragraph.append(content.get_text())
        else:
            # Paragraph or <turn> split

            if content[0] == "\n":
                content_for_paragraph = content[1:]

            else:
                content_for_paragraph = content

            if "\n" in content_for_paragraph:
                segments = [x.strip() for x in content_for_paragraph.split("\n")]
                if len(segments) != 0:
                    if len(tmp_paragraph) != 0:
                        seg = segments.pop(0)
                        if seg != "":
                            tmp_paragraph.append(seg)
                            
                        paragraphs.append(
                            reformat_paragraph(
                                tmp_paragraph, added_events_paragraph, events
                            )
                        )
                    tmp_paragraph = []
                    for seg_id, seg in enumerate(segments):
                        if seg_id == len(segments) - 1 and seg != "":
                            tmp_paragraph.append(seg)
                        else:
                            if seg != "":
                                paragraphs.append([seg])

                    added_events_paragraph = {}

                else:
                    if len(tmp_paragraph) != 0:
                        if tmp_paragraph[-1] == "":
                            tmp_paragraph = tmp_paragraph[:-1]


                        paragraphs.append(
                            reformat_paragraph(
                                tmp_paragraph, added_events_paragraph, events
                            )
                        )
                        tmp_paragraph = []
                        added_events_paragraph = {}
            else:
                tmp_paragraph.append(content_for_paragraph)

            # Sentence split
            tmp = content.replace("\n", " ").strip()

            if tmp != "":
                spacy_doc = nltk.sent_tokenize(tmp)

                # spacy_doc = [x.text for x in nlp(tmp).sents]
                # print(tmp)
                # print(spacy_doc)
                if len(spacy_doc) > 1:
                    # print('='*89)
                    # print("First:", new_content)

                    for s_id, sent in enumerate(spacy_doc):
                        if s_id == 0 and len(tmp_content) != 0:
                            tmp_content.append(sent.strip())

                            new_content.append(
                                reformat_sentence(tmp_content, added_events, events)
                            )
                            tmp_content = []
                            added_events = {}
                        else:
                            if s_id != len(spacy_doc) - 1:
                                if sent in "!?.;":
                                    assert len(tmp_content) == 0
                                    new_content[-1].append(sent)
                                else:
                                    assert len(tmp_content) == 0
                                    new_content.append([sent.strip()])
                            else:
                                if sent[-1] not in "!?.;".split():
                                    tmp_content.append(sent.strip())
                                else:
                                    new_content.append(sent.strip().split())

                else:
                    tmp_content.append(spacy_doc[0])

    if len(tmp_content) != 0:

        new_content.append(reformat_sentence(tmp_content, added_events, events))
        tmp_content = []
        added_events = []

    if len(tmp_paragraph) != 0:
        if tmp_paragraph[-1] == "":
            tmp_paragraph = tmp_paragraph[:-1]
        paragraphs.append(
            reformat_paragraph(tmp_paragraph, added_events_paragraph, events)
        )
        tmp_paragraph = []
        added_events_paragraph = {}

    full_paragraph = []
    full_setences = []
    for k, ev in events.items():
        assert " ".join(new_content[ev.sent_id][ev.offset]) == ev.token
        assert " ".join(paragraphs[ev.paragraph_id][ev.offset_paragraph]) == ev.token
    for par in paragraphs:
        full_paragraph.extend(
            sum(
                [
                    remove_punctuation(x).split()
                    for x in par
                    if remove_punctuation(x) != ""
                ],
                [],
            )
        )
    for sent in new_content:
        full_setences.extend(
            sum(
                [
                    remove_punctuation(x).split()
                    for x in sent
                    if remove_punctuation(x) != ""
                ],
                [],
            )
        )
    error_paragraph = True
    # if full_paragraph != full_tokenized_text:
    #     print(full_paragraph, len(full_paragraph))
    #     print(full_tokenized_text, len(full_tokenized_text))

    differences_paragraph.append(len(full_paragraph) - len(full_tokenized_text))
    differences_sentences.append(len(full_setences) - len(full_tokenized_text))

    assert full_paragraph[-1] == full_tokenized_text[-1]
    assert full_setences[-1] == full_tokenized_text[-1]
    assert len(paragraphs) == n_paragraphs
    assert len(tmp_content) == 0
    assert len(tmp_paragraph) == 0
    assert len(events_in_text) == n_events

    relations = {}
    if d_id in matres:
        print(d_id)
        for rel_ty, rels in matres[d_id].items():
            if rel_ty not in relations:
                relations[rel_ty] = []
            for rel in rels:
                relations[rel_ty].append(
                    Relation(rel_ty, events[rel["eid1"]], events[rel["eid2"]])
                )
        documents[d_id] = Document(
            d_id,
            title,
            extra_info,
            new_content,
            paragraphs,
            "",
            events,
            relations,
            None,
        )
    else:
        print("MISSING:", d_id)

print(Counter(differences_paragraph))
print(Counter(differences_sentences))
output_files = {"train": [], "valid": [], "test": []}

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
    with open("data/TB-DENSE/" + k + ".pkl", "wb") as f:
        pickle.dump(v, f, pickle.HIGHEST_PROTOCOL)
