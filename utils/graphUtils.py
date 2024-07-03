import networkx as nx
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot
from utils.modelUtils import ModelUtils


def create_and_save_graph(model_utils, themed_data, labels, filepath):
    G = nx.Graph()

    # Process each claim and its evidences, assuming they are appropriately labeled
    for idx, group in themed_data.iterrows():
        claim_id = f"Claim_{idx}"
        claim_embeddings = model_utils.get_embeddings([group['Claim_text']])[0]
        G.add_node(claim_id, type='claim', text=group['Claim_text'], embedding=claim_embeddings)

        evidences = group['Evidence_text']
        for i, evidence in enumerate(evidences):
            evidence_id = f"Evidence_{idx}_{i}"
            evidence_embeddings = model_utils.get_embeddings([evidence])[0]
            G.add_node(evidence_id, type='evidence', text=evidence, embedding=evidence_embeddings)
            similarity = cosine_similarity(claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
            if similarity > 0.5:
                G.add_edge(claim_id, evidence_id, weight=similarity)

    # Save the graph
    with open(filepath, "wb") as f:
        pickle.dump(G, f)

    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

def draw_cluster_graph(data, labels, cluster_id, model_utils, title='Cluster Visualization'):
    G = nx.Graph()

    # Add nodes with their respective cluster labels
    for index, row in data.iterrows():
        if labels[index] == cluster_id:
            if 'Claim_text' in row:
                node_id = f"Claim_{index}"
                embedding = model_utils.get_embeddings([row['Claim_text']])[0]
            else:
                node_id = f"Evidence_{index}"
                embedding = model_utils.get_embeddings([row['Evidence_text']])[0]
            
            G.add_node(node_id, type='claim' if 'Claim_text' in row else 'evidence', label=embedding)
    
    # Add edges based on similarity
    for node1, data1 in G.nodes(data=True):
        for node2, data2 in G.nodes(data=True):
            if node1 != node2:
                similarity = cosine_similarity(data1['label'].reshape(1, -1), data2['label'].reshape(1, -1))[0][0]
                if similarity > 0.7:
                    G.add_edge(node1, node2, weight=similarity)

    print(f"Cluster {cluster_id} graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Use Plotly for interactive visualization
    pos = nx.spring_layout(G)
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                     line=dict(width=0.5*edge[2]['weight'], color='blue'),
                                     hoverinfo='none', mode='lines'))

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G],
        y=[pos[node][1] for node in G],
        text=[node for node in G],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[len(G.edges(node)) for node in G],
                    colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')))

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(title=title, showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    plot(fig, filename=f'{title}.html')