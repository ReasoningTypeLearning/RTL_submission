import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

dataset_name = 'reclor'

if dataset_name == 'reclor':
    path = 'dataset/reclor_data/train.json'
elif dataset_name == 'logiqa':
    path = 'dataset/logiqa_data/train.json'

with open(path, 'r') as f:
    data = json.load(f)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus = []
for d in data:
    corpus.append(d['question'])
corpus_embeddings = embedder.encode(corpus)

n_clusters=5
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
clustering_model = KMeans(n_clusters=n_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
centers = clustering_model.cluster_centers_

all_data = np.concatenate((centers, corpus_embeddings), axis=0)
pca = PCA(n_components=2)
pca.fit(all_data)
all_embedded_data = pca.transform(all_data)

center_data, embedded_data = all_embedded_data[:n_clusters, :], all_embedded_data[n_clusters:, :]

plt.figure()
for label in list(range(n_clusters)):
    indices = [index for index, value in enumerate(cluster_assignment) if value==label]
    plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], s=5)

plt.scatter(center_data[:, 0], center_data[:,1], marker='x', color='black')
plt.axis('off')

# Save the image
if dataset_name == 'reclor':
    plt.savefig('reclor_cluster.png')
elif dataset_name == 'logiqa':
    plt.savefig('logiqa_cluster.png')
