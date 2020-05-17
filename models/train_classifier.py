# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pickle



def load_data(database_filepath):
    
    # create engine to connect to the database
    engine = create_engine('sqlite:///' + database_filepath)
    
    # read the table from the database into a dataframe
    df = pd.read_sql("SELECT * FROM InsertTableName", engine)
    
    # create X from the "message" column
    X = df['message']

    # create Y values from the 36 different category columns
    Y = df.iloc[:,4:40]
    
    # take the category names
    category_names = list(Y.columns)

    return X, Y, category_names



def tokenize(text):

    # changing all letters to lowercase
    text = text.lower()
    
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", "", text)
    
    # split text into tokens
    words = word_tokenize(text)
    
    # removing stopwords (for example the, an, my etc.)
    words = [w for w in words if w not in stopwords.words("english")]
    
    # stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # lemmatization, reduce words to their root form.
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed




def build_model():
    
    # define the pipeline with  estimators
    # the last estimator 'clf' has two estimators together for multi-output classification.
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    # define parameters to be tuned with Grid_Search_CV    
    parameters = {
                    #'vect__max_df' : [0.75, 1.0],
                    #'tfidf__use_idf' : [True, False],
                    #'clf__estimator__min_samples_leaf':[1, 10],
                    'clf__estimator__max_depth': [10, None]
                    }
    
    # use Grid_Search_CV for tuning the parameters defined above
    model = GridSearchCV(pipeline, parameters)
        
    return model




def evaluate_model(model, X_test, Y_test, category_names):
    
    # make predictions with the model
    Y_pred = model.predict(X_test)
    
    # print metrics for each category
    for num, col in enumerate(Y_test.columns):
        print(category_names[num], classification_report(Y_test[col], Y_pred[:, num]))
       
    pass




def save_model(model, model_filepath):
    
    # save model to a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))
    
    pass




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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()