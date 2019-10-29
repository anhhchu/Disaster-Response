# Disaster Response Pipeline Project

## Motivation
This project is part of the requirement for Data Scientist Nanodegree on Udacity to follow a Data Analysis end-to-end process including ETL, Machine Learning and Deployment. 

## Project Files

1. Python Scripting files:
- data/process_data.py: Python script to clean, join data in the 2 csv files in the data folder and load data into the sqlite database DisasterResponse.db
- models/train_classifier.py: Python script to load data from Database created from process_data.py, train and test RandomForestClassifier Model to categorize messages using NLTK and GridSearchCV 
- app/run.py: python script to deploy the model onto webapp, where the model outputs the category of each message input in by user

2. Data files: 
- data/disaster_categories.csv: contains 36 categories for each message ID
- data/disaster_messages.csv: contains actual message text
- DisasterResponse.db: sqlite database created from process_data.py

3. Helper files:
- app/go.html, app/master.html: html files to render webapp for run.py output

## Installation/Requirement

The code is written in Python3. To run the Python code, you will need to install necessary packages using `pip install` or `conda install`. 
1. Data Analysis packages: numpy, pandas
2. Database engine: sqlalchemy
3. Natural language processing: NLTK and its dependencies 
4. Machine Learning packages: scikitlearn and its dependencies

### Instructions to run the python script 
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run deploy web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/  or localhost:3001

