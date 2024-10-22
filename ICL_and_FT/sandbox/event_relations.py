import json
from pprint import pprint
import networkx as nx 
import matplotlib.pyplot as plt 
from pyvis.network import Network

net = Network(height="1080px", directed=True)

with open('valid.jsonl') as f:
    dataset = list(f)
with open('train.jsonl') as f:
    dataset += list(f)
from dataclasses import dataclass

@dataclass
class Event():
    id: str
    type: str
    trigger: str
    offset: list
    sent_id: int
    sentence: list

    


def get_event(event, data):

    # event identifies a coreferential chain of events
    # In mention we have all the occurences of an event
    # assert len(event['mention']) == 1

    id_e = event['id']
    type = event['type']
    sent_id = event['mention'][0]['sent_id']
    offset  = tuple(event['mention'][0]['offset'])
    sentence = data['tokens'][sent_id]
    
    trigger = event['mention'][0]['trigger_word']

    assert trigger == " ".join(sentence[int(offset[0]): int(offset[-1])])

    return Event(id_e, type, trigger, offset, sent_id, sentence)

tempToid = {}

tempRelations = {}
labels 
for file in dataset:
    ex = json.loads(file)

    # print(ex['tokens'])

    # pprint(ex['events'])
    events = {}

    for x in ex['events']:
        events[x['id']] = get_event(x, ex)
        
        
    labels = []
    edges = []
    nodes = {}
    ttl = 15
    for k, rels in ex['temporal_relations'].items():
        if k not in tempRelations:
            tempRelations[k] = 0
        if k
        tempRelations[k] += len(rels)
        # for rel in rels:
        #     if ttl <= 0:
        #         break
        #     ttl = ttl -1
        #     if 'EVENT' in rel[0] and 'EVENT' in rel[1]:
        #         n1_sent = [x for x in events[rel[0]].sentence]
        #         n2_sent = [x for x in events[rel[1]].sentence]
        #         n1_sent.insert(events[rel[0]].offset[-1], "</b>")    
        #         n1_sent.insert(events[rel[0]].offset[0], "<b>")       
        #         n2_sent.insert(events[rel[1]].offset[-1], "</b>")    
        #         n2_sent.insert(events[rel[1]].offset[0], "<b>")       
        #         node1 = " ".join(n1_sent)
        #         node2 = " ".join(n2_sent)
        #         if node1 not in nodes:
        #             nodes[node1] = len(nodes)
        #             net.add_node(nodes[node1], label=node1)
        #         if node2 not in nodes:
        #             nodes[node2] = len(nodes)
        #             net.add_node(nodes[node2], label=node2)
        
        #         net.add_edge(nodes[node1], to=nodes[node2], label=k, arrowStrikethrough=True)    
                # nodes.append(events[rel[0]].trigger)
                # nodes.append(events[rel[1]].trigger)
                # edges.append([node1, node2])
                # labels.append(k)

    # TODO: convert nodes to integers from 0 to N, same edges.

pprint(tempRelations)

pprint({k: round(v/sum(tempRelations.values()),3)*100 for k,v in tempRelations.items()})
# G = nx.Graph()
# G.add_edges_from(edges)
# pos = nx.nx_agraph.graphviz_layout(G)
# # plt.figure()
# nx.draw(
#     G, pos, edge_color='black', width=1, linewidths=1,
#     node_size=500, node_color='pink', alpha=0.9,
#     labels={node: node for node in G.nodes()}
# )
# nx.draw_networkx_edge_labels(
#     G, pos,
#     edge_labels={tuple(edge): labels[edge_id] for edge_id, edge in enumerate(edges)},
#     font_color='red'
# )
# # plt.axis('off')
# plt.show()
net.toggle_physics(False)
net.show('mygraph.html', notebook=False)