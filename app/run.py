import json
import plotly
import pandas as pd
import joblib

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from utils.plotting import return_figure
from utils.custom_scorer import multi_output_fscore
from utils.custom_transformer import tokenize, StartingVerbExtractor


app = Flask(__name__)

# load data
database_name = 'DisasterResponse'
engine = create_engine('sqlite:///../data/{}.db'.format(database_name))
df = pd.read_sql_table('{}'.format(database_name), engine)

# load model
model_name = 'classifier'
model = joblib.load("../models/{}.pkl".format(model_name))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    graphs = return_figure(df = df)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls = plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids = ids, graphJSON = graphJSON)


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
    app.run(host = '0.0.0.0', port = 3001, debug = True)


if __name__ == '__main__':
    main()