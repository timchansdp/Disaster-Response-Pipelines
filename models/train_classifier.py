import re
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, f1_score,\
                            precision_recall_fscore_support,\
                            classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

import pickle


def load_data(database_filepath):
    """
    Function to :
        - Load data from database
    
    Args:
        database_filepath (str): File path of database
    
    Returns:
        pandas dataframe: Merged dataframe containing messages and categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    
    indexes = [x.start() for x in re.finditer('/', database_filepath)]
    database_name = database_filepath[indexes[-1] + 1: -3]
    
    df = pd.read_sql_table(database_name, engine)
    X = df["message"]
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
    Function to :
        - process text data
    
    Args:
        text (str): string of messages
    
    Returns:
        clean_tokens (list): list of tokenized text data
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation characters; prefixed with r to indicate that 
    # it is a regular expression
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text / Split text into words using NLTK
    tokens = word_tokenize(text)
    
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Reduce words to their root form and remove white space
    clean_tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in tokens]
            
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """ 
    Class for:
        -extracting whether each sentence started with a verb,
         creating a new feature
            
    """
    
    def starting_verb(self, text):

        sentence_list = nltk.sent_tokenize(text) # tokenize by sentences

        for sentence in sentence_list:
            
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            # Check if pos_tags is empty; true if pos_tags is not empty
            if pos_tags:
                
                # index pos_tags to get the first word and part of speech tag
                first_word, first_tag = pos_tags[0]
                
                # return true if the first word is an appropriate verb 
                # or RT for retweet
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True

        return False

    def fit(self, x, y = None):
        return self

    def transform(self, X):
        
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        
        return pd.DataFrame(X_tagged)


def multi_output_fscore(Y_true, Y_pred):
    """
    Function to :
        - be the input of function make_scorer(), thus being the scoring method 
          of the grid search object created by GridSearchCV()
    
    Args:
        Y_true (pandas dataframe): labels
        Y_pred (pandas dataframe): predictions
        average (str): this determines the type of averaging performed on the 
                       data
    
    Returns:
        fscore_list.mean() (float): mean of f1-score
    """
    fscore_list = []
    
    for i in range(0, Y_true.shape[1]):

        f_score = f1_score(y_true = Y_true.iloc[:, i],\
                           y_pred = Y_pred[:, i],\
                           average = 'weighted',\
                           zero_division = 0)
                            
        fscore_list.append(f_score)

    fscore_list = np.array(fscore_list)

    return fscore_list.mean()


def build_model():
    """
    Function to :
        - Build the model with the parameters selected by grid search
    
    Args:
        None
    
    Returns:
        cv (estimator): machine learning model trained with the parameters 
                        selected by grid search
    """    
    # build pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
    
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
    
            ('starting_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state = 42)))
    ])

    # specify parameters for grid search
    parameters = {
                    'features__text_pipeline__vect__ngram_range': [(1, 2)],\
                    'features__text_pipeline__vect__max_features': [5000],\
                    'clf__estimator__algorithm': ['SAMME.R'],\
                    'clf__estimator__base_estimator': [None],\
                    'clf__estimator__learning_rate': [1],\
                    'clf__estimator__n_estimators': [50, 100]
    }
    
    scorer = make_scorer(multi_output_fscore, greater_is_better = True)

    # create grid search object
    cv = GridSearchCV(estimator = pipeline,\
                      param_grid = parameters,\
                      scoring = scorer,\
                      n_jobs = 1,\
                      refit = True,\
                      cv = 2,\
                      verbose = 4,\
                      error_score = 'raise')
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names, average = 'weighted'):
    """
    Function to :
        - Evaluate model
    
    Args:
        model (estimator): machine learning model
        X_test (pandas series): test data set of X
        Y_test (pandas dataframe): test data set of Y
        category_names (list): name of categories
        average (str): this determines the type of averaging performed 
                       on the data
        
    Returns:
        None
    """
    # Predict test data
    Y_pred = model.predict(X_test)
    
    results = pd.DataFrame(columns = ['Category', 'Precision', 'Recall',\
                                      'F-score'])

    for i in range(len(category_names)):

        category = category_names[i]
        
        precision, recall, f_score, support =\
        precision_recall_fscore_support(Y_test[category],\
                                        Y_pred[:, i],\
                                        average = average,\
                                        zero_division = 0 
        )
        
        results = results.append({'Category': category,\
                                  'Precision': precision,\
                                  'Recall': recall,\
                                  'F-score': f_score},\
                                  ignore_index = True)

    print('Mean Precision:', results['Precision'].mean())
    print('Mean Recall:', results['Recall'].mean())
    print('Mean F_score:', results['F-score'].mean())
    print('\n--------------------Classification Report--------------------\n')
    
    for i in range(len(category_names)):

        category = category_names[i]
        print(category)
        print(classification_report(Y_test[category],\
                                    Y_pred[:, i],\
                                    zero_division = 0))


def save_model(model, model_filepath):
    """
    Function to :
        - Saves model as pickle file
    
    Args:
        model (estimator): machine learning model
        model_filepath (str): path where model will be saved
    
    Returns:
        None
    """
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():

    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        
        print('\n<----- Loading data... ----->\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,\
                                                            test_size = 0.2)
        
        print('\n<----- Building model... ----->')
        model = build_model()
        
        print('\n<----- Training model... ----->')
        model.fit(X_train, Y_train)
        print('\nBest parameters found by grid search:\n{}'.format(model.best_params_))
        
        print('\n<----- Evaluating model... ----->')
        evaluate_model(model, X_test, Y_test, category_names)

        print('\n<----- Saving model... ----->\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('\n\nTrained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()