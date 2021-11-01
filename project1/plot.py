#%%
import csv
from os import name
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
type = "large"
data_file = f"data/{type}.csv"
graph_file = f"{type}.gph"

D = pd.read_csv(data_file)
idx2names = {i: D.columns[i] for i in range(len(D.columns))}
names2idx = {D.columns[i]: i for i in range(len(D.columns))}

G = nx.DiGraph()
with open(graph_file, 'r') as f:
    edges = csv.reader(f, delimiter=",")
    for edge in edges:
        i, j = names2idx[edge[0]], names2idx[edge[1].strip()]
        G.add_edge(i, j)

H = nx.relabel_nodes(G, idx2names)

#%%
# print("Nodes of graph: ")
# print(H.nodes())
# print("Edges of graph: ")
# print(H.edges())
nx.draw_circular(H, with_labels=True, node_size=10, node_color="skyblue", node_shape="o",
        alpha=0.5, linewidths=10, 
        font_size=10, font_color="blue", font_weight="bold", width=2, edge_color="grey")
# plt.figure(1,figsize=(30, 40)) 
plt.savefig(f"graph_{type}.png")
plt.show()
# %%
