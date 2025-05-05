import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class ScientificTextCleaner:
    def __init__(
        self,
        remove_stopwords=True,
        remove_numbers=False,
        remove_punct=True,
        lowercase=True,
        lemmatize=True,
        stem=False,
        min_word_length=2,
        language="english",
        custom_stopwords=None
    ):
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.remove_punct = remove_punct
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_word_length = min_word_length
        self.language = language
        
        # Set up stopwords
        self.stopwords = set(stopwords.words(language))
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            
        # Set up lemmatizer and stemmer
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
            
        if self.stem:
            self.stemmer = PorterStemmer()
            
    def normalize_unicode(self, text):
        # Normalize unicode characters
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    def remove_punctuation(self, text):
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_latex(self, text):
        # Remove LaTeX commands and math environments
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)  # Remove LaTeX commands like \textbf{text}
        text = re.sub(r'\$[^$]*\$', ' ', text)  # Remove inline math
        text = re.sub(r'\\\([^)]*\\\)', ' ', text)  # Remove inline math
        text = re.sub(r'\\\[[^\]]*\\\]', ' ', text)  # Remove display math
        text = re.sub(r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}', ' ', text, flags=re.DOTALL)  # Remove environments
        return text
    
    def remove_citations(self, text):
        # Remove citation markers like [1], [2, 3]
        return re.sub(r'\[\d+(,\s*\d+)*\]', ' ', text)
    
    def remove_urls(self, text):
        # Remove URLs
        return re.sub(r'http[s]?://\S+', ' ', text)
    
    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""
            
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove LaTeX, citations, and URLs
        text = self.remove_latex(text)
        text = self.remove_citations(text)
        text = self.remove_urls(text)
        
        # Lowercase if required
        if self.lowercase:
            text = text.lower()
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Remove short words
            if len(token) < self.min_word_length:
                continue
                
            # Remove stopwords
            if self.remove_stopwords and token in self.stopwords:
                continue
                
            # Remove numbers
            if self.remove_numbers and token.isdigit():
                continue
                
            # Remove punctuation
            if self.remove_punct and all(c in string.punctuation for c in token):
                continue
                
            # Lemmatize
            if self.lemmatize:
                token = self.lemmatizer.lemmatize(token)
                
            # Stem
            if self.stem:
                token = self.stemmer.stem(token)
                
            processed_tokens.append(token)
            
        # Join tokens back into text
        cleaned_text = ' '.join(processed_tokens)
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def clean_dataframe(self, df, text_column='text'):
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Apply cleaning to the text column
        df_clean[text_column] = df_clean[text_column].apply(self.clean_text)
        
        return df_clean
    
    def clean_dataset(self, dataset, text_key='text'):
        # Clean a PyTorch dataset by modifying the text field
        for i in range(len(dataset)):
            sample = dataset[i]
            if text_key in sample:
                sample[text_key] = self.clean_text(sample[text_key])
                dataset[i] = sample
                
        return dataset
    
    def transform(self, sample):
        # Used in the DataLoader's transform pipeline
        if 'text' in sample:
            sample['text'] = self.clean_text(sample['text'])
        return sample 