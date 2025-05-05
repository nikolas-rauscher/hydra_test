from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import gensim
from gensim import corpora, models
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class TopicModelingSentenceTransformerModule(pl.LightningModule):
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        n_components: int = 2,
        random_state: int = 42,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_epsilon: float = 0.2,
        n_topics_per_cluster: int = 1,
        learning_rate: float = 1e-4,
        output_dir: str = "topic_modeling_results",
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        
        # Initialize sentence transformer model
        self.model = SentenceTransformer(embedding_model)
        
        # Store parameters for UMAP and HDBSCAN
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.n_topics_per_cluster = n_topics_per_cluster
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        
        # Initialize results placeholders
        self.embeddings = None
        self.umap_embeddings = None
        self.cluster_labels = None
        self.cluster_analysis = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def preprocess_text(self, text):
        """Prepare text for topic modeling"""
        if not isinstance(text, str):
            return []
            
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        stop_words = set(stopwords.words('english'))
        domain_stops = {'study', 'analysis', 'using', 'based', 'method', 'approach', 'review'}
        stop_words.update(domain_stops)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        return tokens
    
    def create_embeddings(self, texts):
        """Create embeddings for the texts"""
        self.log("status", "Creating embeddings")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def perform_clustering(self, embeddings):
        """Perform UMAP dimensionality reduction and HDBSCAN clustering"""
        self.log("status", "Performing dimensionality reduction with UMAP")
        umap_reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric='cosine',
            random_state=self.random_state
        )
        umap_embeddings = umap_reducer.fit_transform(embeddings)
        
        self.log("status", "Performing clustering with HDBSCAN")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='leaf',
            prediction_data=True
        )
        cluster_labels = clusterer.fit_predict(umap_embeddings)
        
        return umap_embeddings, cluster_labels
    
    def analyze_clusters(self, texts, cluster_labels):
        """Analyze topics in each cluster"""
        self.log("status", "Analyzing clusters")
        unique_clusters = sorted(set(cluster_labels))
        cluster_analysis = {}
        
        # Create DataFrame with texts and cluster labels
        df = pd.DataFrame({
            'text': texts,
            'cluster': cluster_labels
        })
        
        # Cluster statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        
        cluster_analysis["stats"] = {
            "n_clusters": n_clusters,
            "total_documents": len(texts),
            "cluster_sizes": cluster_sizes.to_dict()
        }
        
        # Topic analysis per cluster
        cluster_analysis["clusters"] = {}
        for cluster in unique_clusters:
            if cluster != -1:  # Ignore noise cluster
                cluster_texts = df[df['cluster'] == cluster]['text'].tolist()
                
                cluster_data = {
                    "size": len(cluster_texts),
                    "examples": cluster_texts[:5],
                    "topics": []
                }
                
                processed_texts = [self.preprocess_text(txt) for txt in cluster_texts]
                
                if processed_texts and any(processed_texts):
                    dictionary = corpora.Dictionary(processed_texts)
                    corpus = [dictionary.doc2bow(text) for text in processed_texts]
                    
                    if len(dictionary) > 0:
                        lda_model = models.LdaModel(
                            corpus,
                            num_topics=self.n_topics_per_cluster,
                            id2word=dictionary,
                            passes=20,
                            alpha='auto',
                            random_state=self.random_state
                        )
                        
                        for i in range(self.n_topics_per_cluster):
                            topic = lda_model.show_topic(i, topn=15)
                            cluster_data["topics"].append({
                                "id": i,
                                "terms": [(term, float(prob)) for term, prob in topic]
                            })
                
                cluster_analysis["clusters"][int(cluster)] = cluster_data
        
        return cluster_analysis
    
    def create_visualization(self, umap_embeddings, cluster_labels, output_file):
        """Create visualization of clusters"""
        self.log("status", "Creating visualization")
        plt.figure(figsize=(15, 10))
        
        scatter = sns.scatterplot(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            hue=cluster_labels,
            palette='deep',
            alpha=0.7,
            s=100
        )
        
        plt.title('Document Clusters', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def save_cluster_analysis(self, cluster_analysis, output_file):
        """Save cluster analysis to markdown file"""
        with open(output_file, 'w') as f:
            f.write("# Cluster-Analyse\n\n")
            
            # Cluster statistics
            stats = cluster_analysis["stats"]
            f.write(f"## Statistiken\n")
            f.write(f"- Anzahl der Cluster: {stats['n_clusters']}\n")
            f.write(f"- Gesamtanzahl Dokumente: {stats['total_documents']:,}\n\n")
            
            f.write("## Cluster-Größen\n")
            for cluster, size in stats["cluster_sizes"].items():
                f.write(f"- Cluster {cluster}: {size:,} Dokumente\n")
            f.write("\n")
            
            # Topic analysis per cluster
            f.write("## Themen pro Cluster\n\n")
            clusters = cluster_analysis["clusters"]
            for cluster_id, cluster_data in clusters.items():
                f.write(f"### Cluster {cluster_id} ({cluster_data['size']} Dokumente)\n\n")
                
                f.write("#### Beispiel-Texte:\n")
                for text in cluster_data["examples"]:
                    f.write(f"- {text}\n")
                f.write("\n")
                
                for topic in cluster_data["topics"]:
                    topic_terms = ", ".join([f"{term} ({prob:.3f})" for term, prob in topic["terms"]])
                    f.write(f"#### Hauptthema {topic['id']}:\n{topic_terms}\n\n")
                
                f.write("---\n\n")
        
        return output_file
    
    def forward(self, texts):
        # In this case, forward is just creating the embeddings
        return self.create_embeddings(texts)
    
    def training_step(self, batch, batch_idx):
        # This is not a typical training task, but we use Lightning structure
        # We'll use a dummy loss since we're not actually training a model
        return {"loss": torch.tensor(0.0, requires_grad=True)}
    
    def validation_step(self, batch, batch_idx):
        return {"texts": batch["texts"]}
    
    def validation_epoch_end(self, validation_step_outputs):
        # Collect all texts from validation batches
        all_texts = []
        for output in validation_step_outputs:
            all_texts.extend(output["texts"])
        
        # Create embeddings for all texts
        self.embeddings = self.create_embeddings(all_texts)
        
        # Perform clustering
        self.umap_embeddings, self.cluster_labels = self.perform_clustering(self.embeddings)
        
        # Analyze clusters
        self.cluster_analysis = self.analyze_clusters(all_texts, self.cluster_labels)
        
        # Save results
        visualization_path = os.path.join(self.output_dir, "cluster_visualization.png")
        analysis_path = os.path.join(self.output_dir, "cluster_analysis.md")
        
        self.create_visualization(self.umap_embeddings, self.cluster_labels, visualization_path)
        self.save_cluster_analysis(self.cluster_analysis, analysis_path)
        
        # Log some metrics
        n_clusters = self.cluster_analysis["stats"]["n_clusters"]
        noise_ratio = 0
        if -1 in self.cluster_analysis["stats"]["cluster_sizes"]:
            noise_size = self.cluster_analysis["stats"]["cluster_sizes"][-1]
            total_docs = self.cluster_analysis["stats"]["total_documents"]
            noise_ratio = noise_size / total_docs
        
        self.log("val/n_clusters", n_clusters)
        self.log("val/noise_ratio", noise_ratio)
    
    def test_step(self, batch, batch_idx):
        # Simply return the batch
        return batch
    
    def predict_step(self, batch, batch_idx):
        # For prediction we just return the cluster assignments for new texts
        texts = batch["texts"]
        embeddings = self.create_embeddings(texts)
        
        # We need to use the existing UMAP model in a real implementation
        # Here we'll just simulate by returning random clusters
        # In a complete implementation you'd use the trained models to assign new points
        return {"cluster_labels": np.random.randint(-1, 10, size=len(texts))} 