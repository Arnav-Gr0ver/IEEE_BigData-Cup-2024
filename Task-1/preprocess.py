import networkx as nx
import pandas as pd

# Function to load the graph from train.txt
def load_graph(file_path):
    with open(file_path, 'r') as f:
        num_nodes = int(f.readline().strip())
        edges = [tuple(map(int, line.strip().split())) for line in f]
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    return graph

# Function to calculate Jaccard similarity for a pair of users
def jaccard_similarity(graph, u, v):
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    intersection = neighbors_u & neighbors_v
    union = neighbors_u | neighbors_v
    return len(intersection) / len(union) if len(union) > 0 else 0

# Function to calculate features for all pairs in the test set
def calculate_features(test_file_path, graph):
    test_pairs = pd.read_csv(test_file_path, sep=' ', header=None, names=['source', 'target'])
    test_pairs['feature_1'] = test_pairs.apply(lambda row: jaccard_similarity(graph, row['source'], row['target']), axis=1)
    # Add other features here
    return test_pairs
