import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def create_and_save_graph(model_utils, selected_group, context='instance', theme=None, filepath="claim_evidence_graph.pkl"):
    G = nx.Graph()
    claim = str(selected_group['Claim_text'])
    evidences = selected_group['Evidence_text']
    claim_embeddings = model_utils.get_embeddings([claim])[0]
    G.add_node("Claim_0", embedding=claim_embeddings, type='claim', text=claim)

    if context == 'thematic' and theme:
        G.add_node(theme, type='theme', text=f"Theme: {theme}")
        G.add_edge("Claim_0", theme, weight=1.0, type='theme-connection')

    for i, evidence in enumerate(evidences):
        evidence_embeddings = model_utils.get_embeddings([evidence])[0]
        G.add_node(f"Evidence_0_{i}", embedding=evidence_embeddings, type='evidence', text=evidence)
        similarity = cosine_similarity(claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
        if similarity > 0.5:
            G.add_edge("Claim_0", f"Evidence_0_{i}", weight=similarity)

    with open(filepath, "wb") as f:
        pickle.dump(G, f)

def draw_graph(filepath, context='instance'):
    with open(filepath, "rb") as f:
        G = pickle.load(f)

    plt.figure(figsize=(12, 8))
    plt.gca().set_facecolor('#ebebeb')
    plt.gcf().set_facecolor('#ebebeb')
    pos = nx.spring_layout(G, scale=2)  # Increase the scale for better spread in thematic views

    node_colors = ['pink' if node[1]['type'] == 'claim' else 'lightblue' if node[1]['type'] == 'theme' else 'lightgreen' for node in G.nodes(data=True)]
    node_sizes = [2500 if node[1]['type'] == 'theme' else 2000 for node in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, linewidths=1.5)

    edges = G.edges(data=True)
    weights = [data['weight']*3 if 'weight' in data else 1 for u, v, data in edges]
    edge_colors = ['black' if 'type' in data and data['type'] == 'theme-connection' else 'gray' for u, v, data in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color=edge_colors)

    nx.draw_networkx_labels(G, pos, font_size=10, font_color='#2C3E50', font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_color='black')

    plt.axis('off')
    plt.show()