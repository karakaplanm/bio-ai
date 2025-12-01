"""
Protein-Protein Interaction (PPI) Network Embedding using Node2Vec
This script demonstrates how to create a simple PPI network,
train a Node2Vec model on it, and evaluate the embeddings by checking
similarities between known interacting proteins.

pip install node2vec --break-system-package
"""

from node2vec import Node2Vec
import networkx as nx

# 1. Example PPI graph (tiny)
G = nx.Graph()
edges = [
    ("TP53", "MDM2"),
    ("TP53", "BAX"),
    ("BAX", "CASP3"),
    ("AKT1", "MTOR"),
    ("AKT1", "PIK3CA"),
]
G.add_edges_from(edges)

# 2. Train Node2Vec
node2vec = Node2Vec(G, dimensions=16, walk_length=10, num_walks=50)
model = node2vec.fit()

# 3. Embedding similarity test
sim = model.wv.similarity("TP53", "BAX")
print("Similarity TP53–BAX =", sim)
sim = model.wv.similarity("TP53", "AKT1")
print("Similarity TP53–AKT1 =", sim)