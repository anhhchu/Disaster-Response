import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
import pickle

# libraries for NLTK
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import re

# libraries for ML
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support


def tokenize(text):
    """helper function to process text from message column from dataframe: remove special characters, change text to lower case, etc.
    
    Args:
        text (str): each line of text from the message column 
        
    Returns:
        list: list of cleaned tokens
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def process_text(text):
    """helper function to process text from message column from dataframe: remove special characters, change text to lower case, etc.
    
    Args:
        text (str): each line of text from the message column 
        
    Returns:
        text: clean text
    """ 
    
    text = text.lower()
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"[^a-zA-Z]"," ", text)
    return text

def load_data(database_filepath):
    """Function to load data from sqlite database
    
    Args:
        database_filepath(str): the file location of the cleaned dataframe
        
    Returns:
        X (df): dataframe of message field as feature for the model
        Y (df): dataframe of 36 categories for each message 
        category_names: list of categories in Y
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.DataFrame(pd.read_sql_table('DisasterResponse',engine))
    
    # process text from message column in df
    df['message'] = df['message']
    
    # split to X and Y
    X = df[['message']]
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    # obtain categories list from Y.columns
    categories = Y.columns.tolist()
    

    #print('X shape, Y shape', X.shape, Y.shape)
   
    return X, Y, categories             


def build_model():
    """Function to specify pipeline for vectorizer, tfidf and classifier; as well as define parameters for GridSearchCV
        
    Returns:
        cv (GridSearchCV): model for train and prediction 
    """
   
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize,lowercase=False,stop_words=stop_words)),
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    
    parameters = {
        #'vect__ngram_range':((1,1),(1,2)),
        'tfidf__norm':['l1','l2'],
        #'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators':[10,20],
        'clf__estimator__min_samples_split':[2,4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs = -1, verbose=2)
   
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Function to evaluate test data
    
    Args:
        model: model initiated from load_model()
        X_test (df): dataframe of input data from testset, shape = (n-records,)
        Y_test (df): dataframe of 36 true labels from testset, shape = (n-records, 36) 
        category_names: 36 category names
        
    Returns: None
        Print out Dataframe of accuracy, precision, recall, f1 score fore test data
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    for cat in category_names:
        accuracy = accuracy_score(Y_test[cat], Y_pred_df[cat]) 
        score = precision_recall_fscore_support(Y_test[cat], Y_pred_df[cat], beta=1, average = 'weighted') 

        score_df = pd.DataFrame([[accuracy,score[0],score[1],score[2]]], index = [cat], 
                                columns=['accuracy','precision','recall','f1'])
        print(score_df)


def save_model(model, model_filepath):
    """ Function to save trained model to a pickle file to be deployed to webapp
    
    Args:
        model: model after trained
        model_filepath (str): path to save pickle file
        
    Return:
        pickle file of trained model saved to specified model_file_path
    """
    model_file = open(model_filepath,"wb")
    pickle.dump(model, model_file)
    model_file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X['message'], Y, test_size=0.2)
        #print("X train shape, Y train shape:", X_train.shape, Y_train.shape)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Best parameters set:')
        print(model.best_params_)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
