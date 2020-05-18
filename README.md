# Disaster Response Pipeline Project




### Project Motivation
The project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set that was used contains real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.



### Instructions to Run the Python Scripts
1. Run the following commands in the project's root directory to set up your database and model.


- To run ETL pipeline that cleans data and stores in database
:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`


- To run ML pipeline that trains classifier and saves
 it:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run the following command in the app's directory to run the web app.

`python run.py`


3. Go to http://0.0.0.0:3001/




### File Descriptions
1. process_data.py
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Creates a vocabulary
- Stores the data and vocabulary in a SQLite database

2. train_classifier.py
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. run.py
- Loads data from the SQLite database
- Loads classification model
- Extract data for visualisations 
- Runs the web app