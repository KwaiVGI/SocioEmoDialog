from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import networkx as nx
from scipy.stats import pearsonr
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils_eval.file_utils import load_json

emos = [
    'Joy',                  # 1
    'Satisfaction',         # 2
    'Amusement',            # 3
    'Love',                 # 4
    'Hope',                 # 5
    'Awe',                  # 6
    'Alertness',            # 7
    'Pride',                # 8
    'Gratitude',            # 9
    'Sadness',              # 10
    'Disgust',              # 11
    'Anger',                # 12
    'Anxiety',              # 13
    'Guilt',                # 14
    'Fear',                 # 15
    'Offense',              # 16
    'Embarrassment',        # 17
    'Contempt'              # 18
]

def stats_emotion(json_path):
    emo_category_count = len(emos)
    emo_res = {}
    appeared_emotions = []
    dialogue_count = 0
    sent_count = 0
    
    conversations = load_json(json_path)
    for conversation in conversations.values():
        dialogue_count += 1
        sent_count += conversation["num_utterances"]

        # emotion frequency
        for sentence in conversation["utterances"]:
            emo = sentence["emotion_label"]
            emo_res[emo] = emo_res.get(emo, 0) + 1

        # emotion centrality
        dialogue = conversation["utterances"]
        appeared_emotions_per_dialogue_0 = np.zeros((emo_category_count,), dtype=int)
        appeared_emotions_per_dialogue_1 = np.zeros((emo_category_count,), dtype=int)
        for sentence in dialogue:
            emo = sentence["emotion_label"]
            role = sentence["speaker_id"]
            mapping_emo = emos.index(emo) if emo in emos else 0
            if role == 0:
                appeared_emotions_per_dialogue_0[mapping_emo] = 1
            else:
                appeared_emotions_per_dialogue_1[mapping_emo] = 1
        appeared_emotions.append(appeared_emotions_per_dialogue_0)
        appeared_emotions.append(appeared_emotions_per_dialogue_1)
    
    emo_result = np.stack(appeared_emotions, axis=1)
    draw_centrality(emo_result)
    draw_freq(emo_res)

    print("# dialogues:", dialogue_count)
    print("# utterance:", sent_count)
    print("avg # utterance/dialogue:", sent_count/dialogue_count)


def draw_centrality(matrix):
    num_emotions = matrix.shape[0]
    pearson_matrix = np.zeros((num_emotions, num_emotions))

    for i in range(num_emotions):
        for j in range(num_emotions):
            if i != j:
                r, _ = pearsonr(matrix[i], matrix[j])
                pearson_matrix[i][j] = r

    edge_index_pairs = [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0],
        [1,1,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
    ]
    G = nx.DiGraph()
    for i in range(num_emotions):
        for j in range(i + 1, num_emotions):
            if i!=j and (edge_index_pairs[i][j] == 1 or edge_index_pairs[j][i] == 1):
                weight = pearson_matrix[i][j]
                G.add_edge(emos[i], emos[j], weight=weight)

    pos = {
        # left-negative
        'Anger': (-1, -1),
        'Disgust': (-1, 1),
        'Fear': (-1.5, 1.5),
        'Offense': (-1.5, 0),
        'Anxiety': (-1.5, -1.5),
        'Guilt': (-2, 0),
        'Embarrassment': (-2.5, 1.2),
        'Contempt': (-2.5, -1.2),

        # mid
        'Sadness': (-0.6, 0),
        'Joy': (0, 0.5),
        'Satisfaction': (0, -0.5),

        # right-positive
        'Amusement': (0.5, 0),
        'Love': (1, 0),
        'Hope': (1.5, 0),
        'Gratitude': (1.7, -1.5),
        'Pride': (1.7, 1.5),
        'Awe': (2, 0),
        'Alertness': (2.5, 0),
    }

    # draw edges
    norm = Normalize(vmin=-0.2, vmax=0.2)
    cmap = cm.get_cmap('coolwarm').reversed()
    edge_colors = [cmap(norm(G[u][v]['weight'])) for u, v in G.edges()]

    # draw
    plt.figure(figsize=(18, 8))
    plt.gca().set_facecolor('white')

    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=1000)
    for node, (x, y) in pos.items():
        plt.text(
            x, y - 0.2,
            node,
            fontsize=10,
            ha='center',
            va='top'
        )

    upper_edges = []
    upper_colors = []
    lower_edges = []
    lower_colors = []
    middle_edges = []
    middle_colors = []

    for i, (u, v) in enumerate(G.edges()):
        y_u = pos[u][1]
        y_v = pos[v][1]
        y_avg = (y_u + y_v) / 2
        x_u = pos[u][0]
        x_v = pos[v][0]
        x_avg = (x_u + x_v) / 2

        if (y_u == y_v or x_u == x_v):
            middle_edges.append((u, v))
            middle_colors.append(edge_colors[i])
        elif y_avg > 0 and x_avg < 0 or y_avg < 0 and x_avg > 0 and x_avg < 1.7 or y_avg > 0 and x_avg > 1.7:
            upper_edges.append((u, v))
            upper_colors.append(edge_colors[i])
        else:
            lower_edges.append((u, v))
            lower_colors.append(edge_colors[i])
            
    # Upper edge → Convex upwards
    nx.draw_networkx_edges(
        G, pos,
        edgelist=upper_edges,
        edge_color=upper_colors,
        width=2.0,
        arrows=True,
        connectionstyle='arc3,rad=0.2'
    )

    # Lower edge → Concave downwards
    nx.draw_networkx_edges(
        G, pos,
        edgelist=lower_edges,
        edge_color=lower_colors,
        width=2.0,
        arrows=True,
        connectionstyle='arc3,rad=-0.2'
    )
    # Mid edge
    nx.draw_networkx_edges(
        G, pos,
        edgelist=middle_edges,
        edge_color=middle_colors,
        width=2.0,
        arrows=True,
        connectionstyle='arc3,rad=0'
    )

    # Optional: Display numbers on edges
    # edge_labels = { (u, v): round(G[u][v]['weight'], 4) for u, v in G.edges() }
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.7)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def draw_freq(res):
    labels = ['Joy', 'Love', 'Anxiety', 'Satisfaction', 
            'Alertness', 'Hope', 'Sadness', 'Amusement', 
            'Pride', 'Disgust', 'Anger', 'Gratitude', 
            'Guilt', 'Fear', 'Awe', 'Offense',
            'Embarrassment', 'Contempt']
    women_fre = [34.02, 30.10, 30.31, 25.85,
                23.06, 22.34, 21.30, 15.78, 
                12.23, 11.68, 10.27, 9.18, 
                5.43, 5.68, 5.09, 5.19, 
                4.57, 0.97]
    men_fre =  [36.07, 29.56, 26.13, 30.59,
                27.44, 22.48, 17.12, 18.10, 
                15.73, 10.46, 8.51, 8.33, 
                5.57, 4.49, 5.31, 4.13, 
                4.67, 1.35]
    distribution = [round((i+j)/sum(women_fre+men_fre)*100) for i, j in zip(women_fre, men_fre)]
    
    document = [res.get(k, 0) for k in labels]
    doc_sum = sum(document)
    document_fre = [i/doc_sum*100 for i in document]
    target_fre = [(i+j)/sum(women_fre+men_fre) for i, j in zip(women_fre, men_fre)]

    T = np.array(target_fre[:-1]+[0])
    S = np.array(document[:-1]+[0])
    b = S - T*doc_sum
    A_a = []
    for i, t in enumerate(target_fre):
        tmp = []
        if i == len(target_fre)-1:
            for j in range(len(labels)):
                tmp.append(0) 
        else:
            for j in range(len(labels)):
                if i == j:
                    tmp.append(t-1)
                else:
                    tmp.append(t)
        A_a.append(tmp)
    A = np.array(A_a)
    x = np.linalg.pinv(A)@b
    s, v, d = np.linalg.svd(A)
    y = np.compress(v < 1e-10, d, axis=0)
    jie = x-000*y[0]

    res_fre = []
    add_list = []
    for (i,j) in zip(document, jie):
        j = max(round(j), 0)
        res_fre.append(i+j)
        add_list.append(j)

    print('The script still needs to be added:')
    for (l, a) in zip(labels, add_list):
        print(l,a)

    # draw
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    frequency = distribution
    percentage = document_fre

    datasets = {
        'Real': (frequency, 'lightgray'),
        'SocioEmoDialog': (percentage, 'steelblue'),
    }

    for k in datasets:
        datasets[k] = (datasets[k][0] + [datasets[k][0][0]], datasets[k][1])  # close

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for name, (values, color) in datasets.items():
        ax.plot(angles, values, linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    custom_levels = [ 5, 10, 15, 20]
    visible_levels = custom_levels[:-1]
    ax.set_rgrids(visible_levels, labels=[str(lvl) for lvl in visible_levels], angle=0, color="gray", fontsize=12)
    ax.set_ylim(0, custom_levels[-1] - 0.1)

    ax.spines['polar'].set_visible(False)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, color='black')
    for label, angle in zip(ax.get_xticklabels(), angles):
        label.set_rotation(np.degrees(angle))
        label.set_horizontalalignment('center')

    plt.legend(loc='lower right', bbox_to_anchor=(1.4, -0.05), fontsize=11, frameon=False)

    plt.tight_layout()
    plt.show()


def main():
    project_dir = Path(__file__).resolve().parent.parent.parent
    sample_path = project_dir / 'data/scripts/scripts_sample.json'
    #sample_path = '/Users/if/Desktop/SED_scrpits.jsonl'
    stats_emotion(sample_path)

if __name__ == '__main__':
    main()