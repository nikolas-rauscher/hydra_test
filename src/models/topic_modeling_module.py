from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class TopicModelingModule(pl.LightningModule):
    def __init__(
        self,
        num_topics: int = 10,
        max_features: int = 10000,
        stop_words: str = "english",
        learning_method: str = "online",
        max_iter: int = 10,
        random_state: int = 42,
        n_top_words: int = 20,
        use_tfidf: bool = False,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        # Initialize the vectorizer
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features, 
                stop_words=stop_words
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features, 
                stop_words=stop_words
            )
        
        # Initialize LDA model
        self.lda = LatentDirichletAllocation(
            n_components=num_topics,
            learning_method=learning_method,
            max_iter=max_iter,
            random_state=random_state
        )
        
        # Store feature names for later use
        self.feature_names = None
        
        # Metrics
        self.perplexity = float('inf')
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
        )
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # Extract text from batch
        texts = [sample['text'] for sample in batch]
        
        # Fit vectorizer if it's the first batch
        if batch_idx == 0:
            self.tf_matrix = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            # Transform texts to term frequency matrix
            batch_tf = self.vectorizer.transform(texts)
            self.tf_matrix = torch.cat([self.tf_matrix, batch_tf], dim=0)
        
        # Since sklearn's LDA is not a PyTorch module, we don't calculate gradients
        # We'll fit the LDA model on the entire dataset during validation
        
        # Dummy loss to satisfy PyTorch Lightning
        loss = torch.tensor(0.0, requires_grad=True)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        # Extract text from batch
        texts = [sample['text'] for sample in batch]
        
        # We'll fit the LDA model on the entire dataset during validation_epoch_end
        return {"texts": texts}
    
    def validation_epoch_end(self, outputs):
        # Collect all texts from validation batches
        all_texts = []
        for output in outputs:
            all_texts.extend(output["texts"])
        
        # Fit or update LDA model
        if self.feature_names is None:
            # If not done during training
            tf_matrix = self.vectorizer.fit_transform(all_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            tf_matrix = self.vectorizer.transform(all_texts)
        
        self.lda.fit(tf_matrix)
        
        # Calculate perplexity
        perplexity = self.lda.perplexity(tf_matrix)
        self.perplexity = perplexity
        
        # Log perplexity
        self.log("val_perplexity", perplexity)
        
        # Log top words per topic
        top_words = self.get_top_words()
        for i, words in enumerate(top_words):
            self.logger.experiment.add_text(f"Topic {i} Top Words", ', '.join(words))
        
        # Create and log topic visualizations
        self.visualize_topics(top_words)
        
        return {"val_perplexity": perplexity}
    
    def get_top_words(self):
        n_top_words = self.hparams.n_top_words
        top_words = []
        
        for topic_idx, topic in enumerate(self.lda.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            top_words.append(top_features)
            
        return top_words
    
    def visualize_topics(self, top_words):
        # Create a directory for visualizations if it doesn't exist
        os.makedirs("topic_visualizations", exist_ok=True)
        
        # Create word clouds for each topic
        for topic_idx, words in enumerate(top_words):
            # Create a word frequency dictionary
            word_freq = {word: self.lda.components_[topic_idx, np.where(self.feature_names == word)[0][0]] 
                        for word in words if word in self.feature_names}
            
            # Create a word cloud
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
            
            # Plot the word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f"topic_visualizations/topic_{topic_idx}_wordcloud.png")
            plt.close()
            
            # Log the word cloud to the logger
            self.logger.experiment.add_image(f"Topic {topic_idx} WordCloud", 
                                            np.array(wordcloud.to_array()), 
                                            dataformats="HWC")
    
    def test_step(self, batch, batch_idx):
        # Extract text from batch
        texts = [sample['text'] for sample in batch]
        
        # Transform texts to term frequency matrix
        tf_matrix = self.vectorizer.transform(texts)
        
        # Get topic distributions for the batch
        topic_distributions = self.lda.transform(tf_matrix)
        
        # Calculate perplexity for this batch
        batch_perplexity = self.lda.perplexity(tf_matrix)
        
        # Log perplexity
        self.log("test_perplexity", batch_perplexity)
        
        return {"test_perplexity": batch_perplexity, "topic_distributions": topic_distributions}
    
    def predict_step(self, batch, batch_idx):
        # Extract text from batch
        texts = [sample['text'] for sample in batch]
        
        # Transform texts to term frequency matrix
        tf_matrix = self.vectorizer.transform(texts)
        
        # Get topic distributions for the batch
        topic_distributions = self.lda.transform(tf_matrix)
        
        # Get dominant topics
        dominant_topics = np.argmax(topic_distributions, axis=1)
        
        # Return predictions
        return {
            "topic_distributions": topic_distributions,
            "dominant_topics": dominant_topics
        } 