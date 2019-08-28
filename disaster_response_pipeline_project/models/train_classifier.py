import sys

import nltk
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'stopwords'])

import pickle

import re

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine


def load_data(database_filepath):
    '''
    Loads the data from a given filename to a DataFrame and splits it into the feature variable X and the target variable y.

    :param database_filepath: string - filename of the file to be loaded
    :return: tuple(DataFrame, DataFrame, list(string)) - tuple containing the feature the target variable and the categorie names (X,y,names)
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(engine.table_names()[0], engine)

    X = df.message.values
    y = df.drop(['id', 'message', 'original', 'genre'], 1).values

    return X, y, df.drop(['id', 'message', 'original', 'genre'], 1).columns.values


def tokenize(text):
    # initializing the filter tools
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    Constructing a machine learning pipeline as a model for training ang predicting

    :return: Pipeline - a untrained machine learning pipeline/model
    '''

    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mic', MultiOutputClassifier(RandomForestClassifier()))
    ])


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluating the model by predicting on the test features and comparing the result with the test target variables.
    evaluation results will be printed (precision recall f1-score)

    :param model: Pipeline - a trained model
    :param X_test: DataFrame - the DF holding the test features
    :param Y_test: DataFrame - the DF holding the target variables
    :param category_names: list(string) - a list holding the categories
    :return: none
    :output: printout of the models performance
    '''

    # predicting on the test features
    y_pred = model.predict(X_test)

    # creating two Data Frames for easy handling
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    y_test_df = pd.DataFrame(Y_test, columns=category_names)

    # prontout with leading category name
    for col in y_pred_df.columns:
        print(col)
        print(classification_report(y_test_df[col].values, y_pred_df[col].values))


def save_model(model, model_filepath):
    '''
    saves the given model under the given name as a pickle file

    :param model: Pipeline - a machine learning model
    :param model_filepath: string - file name for the pickle file
    :return: none
    '''

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()