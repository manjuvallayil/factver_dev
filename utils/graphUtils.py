import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def create_and_save_graph(model_utils, selected_group, filepath="claim_evidence_graph.pkl"):
    G = nx.Graph()
    claim = str(selected_group['Claim_text'])
    evidences = selected_group['Evidence_text']
    claim_embeddings = model_utils.get_embeddings([claim])[0]
    G.add_node("Claim_0", embedding=claim_embeddings, type='claim', text=claim)

    for i, evidence in enumerate(evidences):
        evidence_embeddings = model_utils.get_embeddings([evidence])[0]
        similarity = cosine_similarity(claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
        if similarity > 0.5:
            G.add_edge("Claim_0", f"Evidence_0_{i}", weight=similarity)

    with open(filepath, "wb") as f:
        pickle.dump(G, f)

def draw_graph(filepath):
    with open(filepath, "rb") as f:
        G = pickle.load(f)

    plt.figure(figsize=(12, 8))
    plt.gca().set_facecolor('#ebebeb')
    plt.gcf().set_facecolor('#ebebeb')
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='pink', linewidths=1.5)#edgecolors='black'
    edges = G.edges(data=True)
    weights = [data['weight'] * 3 for u, v, data in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='#2C3E50', font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_color='black')

    plt.axis('off')
    plt.show()
