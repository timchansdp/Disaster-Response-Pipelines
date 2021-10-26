import re
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):
    
    text = text.lower() # Convert to lowercase
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) # Remove punctuation characters; prefixed with r to indicate that it is a regular expression
    
    tokens = word_tokenize(text) # tokenize text / Split text into words using NLTK
    
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    clean_tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in tokens] # Reduce words to their root form and remove white space
            
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):

        sentence_list = nltk.sent_tokenize(text) # tokenize by sentences

        for sentence in sentence_list:

            pos_tags = nltk.pos_tag(tokenize(sentence)) # tokenize each sentence into words and tag part of speech

            if pos_tags: # Check if pos_tags is empty; true if pos_tags is not empty

                first_word, first_tag = pos_tags[0] # index pos_tags to get the first word and part of speech tag

                if first_tag in ['VB', 'VBP'] or first_word == 'RT': # return true if the first word is an appropriate verb or RT for retweet
                    return True

        return False


    def fit(self, x, y = None):
        return self


    def transform(self, X):
        
        X_tagged = pd.Series(X).apply(self.starting_verb) # apply starting_verb function to all values in X
        
        return pd.DataFrame(X_tagged)