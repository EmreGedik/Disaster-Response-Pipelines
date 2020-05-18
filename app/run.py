import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesAndLabels', engine)
vocabulary_df = pd.read_sql_table('Vocabulary', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extracting data for my second visualisation
    occurances = vocabulary_df.groupby('word').sum()['occurance'].nlargest(10)
    most_common_words = list(occurances.index)
    
    # extracting data for my third visualisation
    most_common_label_counts = df.iloc[:,4:40].sum(axis=0, skipna = True).sort_values(ascending=False).nlargest(10)
    label_names = list(most_common_label_counts.index)
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # start of second visualisation data and layout
                {
            'data': [
                Bar(
                    x=most_common_words,
                    y=occurances
                )
            ],

            'layout': {
                'title': 'Most Common Words in the Vocabulary',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        
        # start of third visualisation data and layout
                {
            'data': [
                Bar(
                    x=label_names,
                    y=most_common_label_counts
                )
            ],

            'layout': {
                'title': 'Most Common Labels in the Dataset',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Labels"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()