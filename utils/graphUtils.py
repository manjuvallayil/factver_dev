import networkx as nx
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot

def create_and_save_graph(model_utils, themed_data, filepath):
    G = nx.Graph()

    # Process each claim and its evidences, assuming they are appropriately labeled
    for idx, group in themed_data.iterrows():
        unique_id = group['Claim_topic_id'].split('_')[-1]
        claim_id = f"Claim_{unique_id}"
        claim_embeddings = model_utils.get_embeddings([group['Claim_text']])[0]
        G.add_node(claim_id, type='claim', text=group['Claim_text'], embedding=claim_embeddings)

        evidences = group['Evidence_text']
        for i, evidence in enumerate(evidences):
            evidence_id = f"Evidence_{unique_id}_{i}"
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
            unique_id = row['Claim_topic_id'].split('_')[-1]
            if 'Claim_text' in row:
                node_id = f"Claim_{unique_id}"
                embedding = model_utils.get_embeddings([row['Claim_text']])[0]
                G.add_node(node_id, type='claim', label=embedding)
            for i, evidence in enumerate(row['Evidence_text']):
                evidence_id = f"Evidence_{unique_id}_{i}"
                embedding = model_utils.get_embeddings([evidence])[0]
                G.add_node(evidence_id, type='evidence', label=embedding)

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
                                     hoverinfo='none', mode='lines', showlegend=False))

    node_colors = []
    for node in G:
        if G.nodes[node]['type'] == 'claim':
            node_colors.append('coral')
        else:
            node_colors.append('teal')

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G],
        y=[pos[node][1] for node in G],
        text=[node for node in G],  # Only show IDs
        mode='markers+text',
        hoverinfo='text',
        marker=dict(showscale=True, color=node_colors, size=15,
                    colorbar=dict(thickness=15, title='Node Type', xanchor='left', titleside='right')),
        showlegend=False)

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(title=title, showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    plot(fig, filename=f'{title}.html')
"""
def draw_interconnections(data, labels, cluster_id, selected_claim_id, model_utils, title='Interconnections Visualization', similarity_threshold=0.7):
    G = nx.Graph()

    # Find the selected claim and its related evidences
    selected_claim = None
    for index, row in data.iterrows():
        if labels[index] == cluster_id:
            unique_id = row['Claim_topic_id'].split('_')[-1]
            if f"Claim_{unique_id}" == selected_claim_id:
                selected_claim = row
                break

    if selected_claim is None:
        print(f"No claim found with ID {selected_claim_id} in cluster {cluster_id}")
        return

    claim_text = selected_claim['Claim_text']
    claim_embeddings = model_utils.get_embeddings([claim_text])[0]
    G.add_node(selected_claim_id, type='claim', embedding=claim_embeddings, cluster_id=cluster_id, color='#FA8072', size=35)

    evidences = selected_claim['Evidence_text']
    for i, evidence in enumerate(evidences):
        evidence_id = f"Evidence_{selected_claim_id.split('_')[-1]}_{i}"
        evidence_embeddings = model_utils.get_embeddings([evidence])[0]
        G.add_node(evidence_id, type='evidence', embedding=evidence_embeddings, cluster_id=cluster_id, color='#20B2AA', size=35)
        similarity = cosine_similarity(claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
        if similarity > similarity_threshold:
            G.add_edge(selected_claim_id, evidence_id, weight=similarity)

    # Check for evidence nodes from different clusters and add them with different colors
    for index, row in data.iterrows():
        if labels[index] != cluster_id:
            unique_id = row['Claim_topic_id'].split('_')[-1]
            evidence_texts = row['Evidence_text']
            for i, evidence in enumerate(evidence_texts):
                evidence_id = f"Evidence_{unique_id}_{i}"
                evidence_embeddings = model_utils.get_embeddings([evidence])[0]
                G.add_node(evidence_id, type='evidence', embedding=evidence_embeddings, cluster_id=labels[index], color='#008080', size=10)
                similarity = cosine_similarity(claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
                if similarity > similarity_threshold:
                    G.add_edge(selected_claim_id, evidence_id, weight=similarity)
            # Add the related claim with a different color
            related_claim_id = f"Claim_{unique_id}"
            related_claim_embeddings = model_utils.get_embeddings([row['Claim_text']])[0]
            G.add_node(related_claim_id, type='claim', embedding=related_claim_embeddings, cluster_id=labels[index], color='#FA8072', size=25)
            similarity = cosine_similarity(claim_embeddings.reshape(1, -1), related_claim_embeddings.reshape(1, -1))[0][0]
            if similarity > similarity_threshold:
                G.add_edge(selected_claim_id, related_claim_id, weight=similarity)

    print(f"Interconnections graph for {selected_claim_id} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Remove isolated nodes (nodes without edges)
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(isolated_nodes)

    # Use Plotly for interactive visualization
    pos = nx.spring_layout(G, k=0.15)  # Increase spacing between nodes
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                     line=dict(width=0.5*edge[2]['weight'], color='#808080'),
                                     hoverinfo='none', mode='lines',
                                     showlegend=False))
        
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G],
        y=[pos[node][1] for node in G],
        text=[node for node in G],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(showscale=True,
                    color=[G.nodes[node]['color'] for node in G],
                    size=[G.nodes[node]['size'] for node in G],
                    colorbar=dict(thickness=15, title='Node Type', xanchor='left', titleside='right')),
        showlegend=False)

    fig = go.Figure(data=edge_trace + [node_trace],
                layout=go.Layout(title=title, hovermode='closest',
                                 margin=dict(b=0, l=0, r=0, t=40),
                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    # Add legend manually
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='#FA8072'),
        legendgroup='Central Claim',
        showlegend=True,
        name='Central Claim'
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='#20B2AA'),
        legendgroup='Direct Evidence',
        showlegend=True,
        name='Direct Evidence'
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='#008080'),
        legendgroup='Related Evidence',
        showlegend=True,
        name='Related Evidence'
    ))

    plot(fig, filename=f'{title}.html')
    """

def draw_interconnections(data, labels, cluster_id, selected_claim_id, model_utils, title='Interconnections Visualization', similarity_threshold=0.7):
    G = nx.Graph()

    # Find the selected claim and its related evidences
    selected_claim = None
    for index, row in data.iterrows():
        if labels[index] == cluster_id:
            unique_id = row['Claim_topic_id'].split('_')[-1]
            if f"Claim_{unique_id}" == selected_claim_id:
                selected_claim = row
                break

    if selected_claim is None:
        print(f"No claim found with ID {selected_claim_id} in cluster {cluster_id}")
        return

    claim_text = selected_claim['Claim_text']
    claim_embeddings = model_utils.get_embeddings([claim_text])[0]
    G.add_node(selected_claim_id, type='claim', embedding=claim_embeddings, cluster_id=cluster_id, color='#FA8072', size=35)

    # Add direct evidences of the selected claim
    evidences = selected_claim['Evidence_text']
    for i, evidence in enumerate(evidences):
        evidence_id = f"Evidence_{selected_claim_id.split('_')[-1]}_{i}"
        evidence_embeddings = model_utils.get_embeddings([evidence])[0]
        G.add_node(evidence_id, type='evidence', embedding=evidence_embeddings, cluster_id=cluster_id, color='#20B2AA', size=30)
        similarity = cosine_similarity(claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
        if similarity > similarity_threshold:
            G.add_edge(selected_claim_id, evidence_id, weight=similarity)

    # Add other evidences and claims in the same cluster
    for index, row in data.iterrows():
        if labels[index] == cluster_id:
            unique_id = row['Claim_topic_id'].split('_')[-1]
            if f"Claim_{unique_id}" != selected_claim_id:
                related_claim_id = f"Claim_{unique_id}"
                related_claim_embeddings = model_utils.get_embeddings([row['Claim_text']])[0]
                G.add_node(related_claim_id, type='claim', embedding=related_claim_embeddings, cluster_id=labels[index], color='#FA8072', size=30)
                similarity = cosine_similarity(claim_embeddings.reshape(1, -1), related_claim_embeddings.reshape(1, -1))[0][0]
                if similarity > similarity_threshold:
                    G.add_edge(selected_claim_id, related_claim_id, weight=similarity)

            evidence_texts = row['Evidence_text']
            for i, evidence in enumerate(evidence_texts):
                evidence_id = f"Evidence_{unique_id}_{i}"
                evidence_embeddings = model_utils.get_embeddings([evidence])[0]
                if evidence_id not in G:
                    G.add_node(evidence_id, type='evidence', embedding=evidence_embeddings, cluster_id=labels[index], color='#20B2AA', size=15)
                    similarity = cosine_similarity(claim_embeddings.reshape(1, -1), evidence_embeddings.reshape(1, -1))[0][0]
                    if similarity > similarity_threshold:
                        G.add_edge(selected_claim_id, evidence_id, weight=similarity)

    print(f"Interconnections graph for {selected_claim_id} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Remove isolated nodes (nodes without edges)
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(isolated_nodes)

    # Use Plotly for interactive visualization
    pos = nx.spring_layout(G, k=0.15)  # Increase spacing between nodes
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                     line=dict(width=0.5*edge[2]['weight'], color='#808080'),
                                     hoverinfo='none', mode='lines',
                                     showlegend=False))

    node_trace = go.Scatter(
        x=[pos[node][0] for node in G],
        y=[pos[node][1] for node in G],
        text=[node for node in G],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(showscale=True,
                    color=[G.nodes[node]['color'] for node in G],
                    size=[G.nodes[node]['size'] for node in G],
                    colorbar=dict(thickness=15, title='Node Type', xanchor='left', titleside='right')),
        showlegend=False)

    fig = go.Figure(data=edge_trace + [node_trace],
                layout=go.Layout(title=title, hovermode='closest',
                                 margin=dict(b=0, l=0, r=0, t=40),
                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    # Add legend manually
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='#FA8072'),
        legendgroup='Central Claim',
        showlegend=True,
        name='Central Claim'
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='#20B2AA'),
        legendgroup='Direct Evidence',
        showlegend=True,
        name='Direct Evidence'
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='#008080'),
        legendgroup='Related Evidence',
        showlegend=True,
        name='Related Evidence'
    ))

    plot(fig, filename=f'{title}.html')