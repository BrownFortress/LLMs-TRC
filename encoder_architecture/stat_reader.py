import json
import numpy as np
import pickle
import os
from sandbox.data_structures import *
from utils.dataloader import Prompt, DataRow
from pprint import pprint
from collections import Counter
from sklearn.metrics import f1_score


@dataclass
class DataRowMerge:
    d_id: str
    label: str
    context: Document
    event1: str
    event2: str
    predictions: list
    prediction: str


def has_numbers(input_string) -> bool:
    "It checks if a string has a numebr or not"
    return any(char.isdigit() for char in input_string)

def get_result_by_id(key, res):
    for x in res:
        if x['id'] == key:
            return x

def is_there_timex(event1, event2, document, verbose=False) -> bool:
    "It checks if there is a timex in sentences of e1 and e2"
    s_ids = [event1.sent_id, event2.sent_id]

    for _, timex in document.time_expressions.items():
        if timex.sent_id in s_ids:
            if verbose:
                print("E1", event1.token)
                print("E2", event2.token)
                print("TIMEX", timex.token)
                for x in set(s_ids):
                    print(document.text[x])
            if has_numbers(timex.token):
                return True
    return False


def majority_voting(list_data_row_merge, verbose=False) -> list[DataRowMerge]:
    """Function for computing majority voting over runs"""
    gt = []
    predictions = []
    n_classes = len(set([item.label for _, item in list_data_row_merge.items()]))
    n_runs = len(list(samples.values())[0].predictions)
    # print('Classes', n_classes, 'runs', n_runs)
    new_samples = []
    for _, item in list_data_row_merge.items():
        most_predicted = Counter(item.predictions).most_common(1)[0]
        tmp_sample = item
        tmp_sample.prediction = most_predicted[0]
        new_samples.append(tmp_sample)
        predictions.append(most_predicted[0])
        gt.append(item.label)
        assert most_predicted[1] > round(
            n_runs / n_classes
        )  # TODO: implement a strategy if this fails
    if verbose:
        print(
            "Accuracy Majority Voting:",
            round(f1_score(gt, predictions, average="micro"), 3),
        )
    return new_samples


results = json.load(open("outputs/results.json", "r"))
dump_path = os.path.join("outputs", "dump_results.pkl")

# with open(dump_path, "rb") as reader:
#     dump = pickle.load(reader)
hard_coded_classes = ["BEFORE", "AFTER", "EQUAL"]
print("CLASSES ARE HARD CODED !!", hard_coded_classes)
data = {}
for elm in results:
    model_name = elm["config"]["model_name"]
    if elm["config"]["cls_only"]:
        model_name += " CLS "
    else:
        model_name += " word_level "

    if elm["config"]["init_weights"]:
        model_name += " rand_init "

    if elm["config"]["dual_optimizer"]:
        model_name += " dual_optimizer "

    # if elm["config"]["frozen"]:
    #     model_name += " frozen "

    if "only_context_intra" in elm["config"].keys():
        if elm["config"]["only_context_intra"]:
            model_name += " intra_only "

    if "frozen" in elm["config"].keys():
        if elm["config"]["frozen"]:
            model_name += " frozen "
    model_name += " lr " + str(elm["config"]["lr1"]) + " "
    if len(elm["config"]["layers"]) > 1:
        model_name += " MLP "
    if "context_size" in elm["config"]:
        model_name += " context size " + str(elm["config"]["context_size"]) + " "
    if "cls_as_context" in elm['config']:
        if elm['config']["cls_as_context"]:
            model_name += " cls_as_context "
    if 'dataset_name' in elm["config"]:
        model_name += " " + elm["config"]["dataset_name"] + " "
        
    model_name += elm["created_at"]
    
    if "accuracy" in elm:
        print(model_name)
        data[elm["id"]] = model_name
        print("\tAVG", round(np.asanyarray(elm["accuracy"]).mean(), 3) * 100)
        print("\tSTD", round(np.asanyarray(elm["accuracy"]).std(), 3) * 100)
    per_class = {}
    for k in elm:
        if k in ["BEFORE", "AFTER", "EQUAL"]:  # TODO make it not hard coded
            if k not in per_class:
                per_class[k] = []
            for rep in elm[k]:
                per_class[k].append(rep["f1-score"])

    row_result = ""
    header = ""
    if "accuracy" in elm:
        for k, v in per_class.items():
            header += "\t" + k
            row_result += "\t" + str(round(np.asanyarray(v).mean(), 3) * 100)
        print(header)
        print(row_result)
        print(row_result.replace("\t", ","))


DATABASE_PATH = "data/MATRES"
BASE_MODEL_PATH = "bin"
splits = {}
for file in os.listdir(DATABASE_PATH):
    root_name = file.split(".")[0]
    if root_name in ["train", "valid", "test"]:
        with open(os.path.join(DATABASE_PATH, file), "rb") as reader:
            splits[root_name] = pickle.load(reader)

events_in_training = []

for doc in splits["train"]:
    d_id = doc.d_id
    for rel_type, rels in doc.temporal_relations.items():
        if rel_type != "VAGUE":
            for rel in rels:
                event1 = rel.event1
                event2 = rel.event2
                events_in_training.append((event1.token, event2.token, rel_type))

wrong_predictions = {}
correct_predictions = {}

for elm in dump:
    errors = {}
    standardise = {}
    if elm["id"] in data:
        # if "CLS" in data[elm["id"]]:
        print(data[elm["id"]])
        res = get_result_by_id(elm["id"], results)
        model_key = res['config']['model_name'] + elm["id"]
        
        wrong_predictions[model_key] = []
        correct_predictions[model_key] = []

        for run in elm["data"]:
            for row in run:
                if row.prediction != row.label:
                    if row.label not in errors:
                        errors[row.label] = {}
                    distance = abs(row.event1.sent_id - row.event2.sent_id)
                    if distance not in errors[row.label]:
                        errors[row.label][distance] = 0
                    errors[row.label][distance] += 1
                    wrong_predictions[model_key].append(
                        (row.event1.token, row.event2.token, row.label, row.context.d_id)
                    )
                    # print("SAM SENT:", "E1:",row.event1.token, "E2:" ,row.event2.token, "Rel:",  row.label, 'Pred:', row.prediction)
                    # print("\t",  " ".join(row.context.text[row.event1.sent_id]), row.event1.sent_id)
                    # print("\t", " ".join(row.context.text[row.event2.sent_id]), row.event2.sent_id)
                else:
                     correct_predictions[model_key].append((row.event1.token, row.event2.token, row.label))
                if row.label not in standardise:
                    standardise[row.label] = []
                standardise[row.label].append(row.prediction)

    # pprint({k:v for k,v in errors.items()})
    pprint(
        {
            lab: {
                k: round(v / len(standardise[lab]), 3) * 100 for k, v in counts.items()
            }
            for lab, counts in errors.items()
        }
    )

events_in_training_tokens_pairs = []
events_in_training_tokens_pairs_set = []
events_in_training_tokens = []

for pair in events_in_training:
    events_in_training_tokens_pairs_set.append({pair[0], pair[1]})
    events_in_training_tokens_pairs.append((pair[0], pair[1]))
    events_in_training_tokens.extend([pair[0], pair[1]])

all_wrong_predictions = {}
for k, events in wrong_predictions.items():
    print(k)
    unseen_pairs_set = []
    unseen_pairs = []
    unseen_example = []
    unseen_tokens = []
    all_wrong_predictions[k] = []
    for ev in events:
        e = ev[0:-1]
        all_wrong_predictions[k].append(ev)
        # if e not in events_in_training:
        #     unseen_example.append(e)
        # if (e[0], e[1]) not in events_in_training_tokens_pairs:
        #     unseen_pairs.append((e[0], e[1]))
        # if {e[0], e[1]} not in events_in_training_tokens_pairs_set:
        #     unseen_pairs_set.append({e[0], e[1]})
        # if e[0] not in events_in_training_tokens:
        #     unseen_tokens.append(e[0])
        # if e[1] not in events_in_training_tokens:
        #     unseen_tokens.append(e[1])
    # print("="*10, "Wrong", "="*10)
    # print("\t Unseen examples:", round(len(unseen_example)/len(events),3)*100)
    # print("\t Unseen pairs:", round(len(unseen_pairs)/len(events),3)*100 )
    # print("\t \t  Unseen pairs as a set:", round(len(unseen_pairs_set)/len(events),3)*100)  
    # print("\t Unseen tokens:", round(len(unseen_tokens)/(len(events)*2),3)*100) 
    unseen_pairs_set = []
    unseen_pairs = []
    unseen_example = []
    unseen_tokens = []
    # for e in correct_predictions[k]:
    #     if e not in events_in_training:
    #         unseen_example.append(e)
    #     if (e[0], e[1]) not in events_in_training_tokens_pairs:
    #         unseen_pairs.append((e[0], e[1]))
        # if {e[0], e[1]} not in events_in_training_tokens_pairs_set:
        #     unseen_pairs_set.append({e[0], e[1]})
        # if e[0] not in events_in_training_tokens:
        #     unseen_tokens.append(e[0])
        # if e[1] not in events_in_training_tokens:
        #     unseen_tokens.append(e[1])
    # print("="*10, "CORRECT", "="*10)
    # print("\t Unseen examples:", round(len(unseen_example)/len(correct_predictions[k]),3)*100)
    # print("\t Unseen pairs:", round(len(unseen_pairs)/len(correct_predictions[k]),3)*100 )
    # print("\t \t  Unseen pairs as a set:", round(len(unseen_pairs_set)/len(correct_predictions[k]),3)*100)  
    # print("\t Unseen tokens:", round(len(unseen_tokens)/(len(correct_predictions[k])*2),3)*100)

list_of_wrong_predicitons = [set(v) for k, v in all_wrong_predictions.items() if "roberta-base" in k]

print(len(set.intersection(*list_of_wrong_predicitons)))


# time_expression_influence = {}
# documents = {}
# samples_per_model = {}


# for elm in dump:
#     errors = {}
#     standardise = {}
#     print(data[elm["id"]])
#     samples = {}
#     for run in elm["data"]:
#         for row in run:
#             doc = row.context
#             if doc.d_id not in documents:
#                 documents[doc.d_id] = doc
#             e1 = row.event1
#             e2 = row.event2
#             key = (
#                 doc.d_id,
#                 e1.token,
#                 str(e1.sent_id),
#                 str(e1.offset.start),
#                 e2.token,
#                 str(e2.sent_id),
#                 str(e2.offset.start),
#             )
#             if key not in samples:
#                 samples[key] = DataRowMerge(
#                     d_id=row.d_id,
#                     label=row.label,
#                     context=row.context,
#                     event1=row.event1,
#                     event2=row.event2,
#                     predictions=[],
#                     prediction=None
#                 )
#             samples[key].predictions.append(row.prediction)
#     samples_per_model[data[elm["id"]]] = majority_voting(samples, verbose=True)

# statistics = {"errors":[], "correct": []}
# intra_inter = {"intra":[], "inter": []}
# for model_id, samples in samples_per_model.items():
#     print(model_id)
#     for sample in samples:
#         d_id = sample.context.d_id
#         context = sample.context
#         e1 = sample.event1
#         e2 = sample.event2
#         gt = sample.label
#         pred = sample.prediction

#         if e1.sent_id == e2.sent_id:
#             intra_inter["intra"].append("correct" if gt == pred else "wrong")
#         else:
#             intra_inter["inter"].append("correct" if gt == pred else "wrong")


#         time_expression = "Yes" if is_there_timex(e1,e2, context, verbose=False) else "No"

#         if gt != pred:
#             statistics["errors"].append(time_expression)
#         else:
#             statistics["correct"].append(time_expression)

#     N_EXAMPLES = len(statistics["correct"]) + len(statistics["errors"])

#     pprint(
#         {
#             lab: {
#                 k: round(v / len(values), 3) * 100 for k, v in Counter(values).items()
#             }
#             for lab, values in statistics.items()
#         }
#     )


#     pprint(
#         {
#             lab: {
#                 k: round(v / len(values), 3) * 100 for k, v in Counter(values).items()
#             }
#             for lab, values in intra_inter.items()
#         }
#     )
