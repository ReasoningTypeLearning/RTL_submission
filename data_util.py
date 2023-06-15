from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import pickle
import re, random, json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np

class QuestionAnswerDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, n_clusters, phase):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.phase = phase
        with open(path, 'r') as f:
            self.data = json.load(f)

        if self.phase == 'train':
            self.cluster_assignment = self.clustering(n_clusters).tolist()
            # for baseline
            # random.shuffle(self.cluster_assignment)
            num_each_cluster = {}
            for i in range(n_clusters):
                num_each_cluster[i] = 0
            for item in self.cluster_assignment:
                num_each_cluster[item] +=1
            print('qtype label assignment:',num_each_cluster)
    
    def __len__(self):
        return len(self.data)
    
    def clustering(self, n_clusters):
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        corpus = []
        for d in self.data:
            corpus.append(d['question'])
        corpus_embeddings = embedder.encode(corpus)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) 

        # clustering_model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=None)
        clustering_model = KMeans(n_clusters=n_clusters, random_state=192)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        return cluster_assignment

    def __getitem__(self, index: int):
        this_data = self.data[index]
        context = this_data['context']
        question = this_data['question']
        answers = this_data['answers']
        
        num_options = len(answers)
        cqa = []
        for an in answers:
            cqa.append(context + '</s>' + question + ' ' + an)
        encoded_cqa = self.tokenizer(cqa, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")

        if self.phase == 'train':
            ca = context + '</s>' + answers[this_data['label']]
            encoded_ca = self.tokenizer(ca, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
            _type = self.cluster_assignment[index]
            # _type = this_data['q_type']

            return dict(
                    src=this_data, 
                    tgt=torch.tensor(this_data['label']),
                    cqa_input_ids=encoded_cqa["input_ids"],
                    cqa_attention_mask=encoded_cqa["attention_mask"],
                    ca_input_ids=encoded_ca["input_ids"].flatten(),
                    ca_attention_mask=encoded_ca["attention_mask"].flatten(),
                    q_type=torch.tensor(_type),
                    )

        elif self.phase == 'val':
            return dict(
                    src=this_data, 
                    tgt=torch.tensor(this_data['label']),
                    cqa_input_ids=encoded_cqa["input_ids"],
                    cqa_attention_mask=encoded_cqa["attention_mask"],
                    )

        else:
            try:
                return dict(
                        src=this_data,
                        cqa_input_ids=encoded_cqa["input_ids"],
                        cqa_attention_mask=encoded_cqa["attention_mask"],
                        tgt=torch.tensor(this_data['label']), # for logiqa
                        )
            except:
                return dict(
                        src=this_data,
                        cqa_input_ids=encoded_cqa["input_ids"],
                        cqa_attention_mask=encoded_cqa["attention_mask"],
                        )


class QuestionAnswerDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size, path, n_clusters, max_len=256):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = QuestionAnswerDataset(path+'train.json', tokenizer, max_len, n_clusters, 'train')
        self.dev_dataset = QuestionAnswerDataset(path+'val.json', tokenizer, max_len, n_clusters, 'val')
        self.test_dataset = QuestionAnswerDataset(path+'test.json', tokenizer, max_len, n_clusters, 'test')      
    
    def __len__(self):
        return len(self.train_dataset)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

def set_hyerparameters(model_name, dataset_name, scheduler):
    # defalut
    co_q = 0.1
    n_clusters = 5

    # After hyerparameters tuning
    if model_name == 'roberta-large':
        if dataset_name == 'reclor':
            if scheduler == 'increment':
                n_clusters = 26
            elif scheduler == 'recurrent':
                n_clusters = 30
            elif scheduler == 'random':
                n_clusters = 24
        elif dataset_name == 'logiqa':
            if scheduler == 'increment':
                n_clusters = 20
            elif scheduler == 'recurrent':
                n_clusters = 13
            elif scheduler == 'random':
                n_clusters = 13
    elif model_name == 'chitanda/merit-roberta-large-v2':
        if dataset_name == 'reclor':
            if scheduler == 'increment':
                n_clusters = 3
                co_q = 0.3
            elif scheduler == 'recurrent':
                n_clusters = 34
            elif scheduler == 'random':
                n_clusters = 34
        elif dataset_name == 'logiqa':
            if scheduler == 'increment':
                n_clusters = 29
            elif scheduler == 'recurrent':
                n_clusters = 38
            elif scheduler == 'random':
                n_clusters = 26

    return co_q, n_clusters
