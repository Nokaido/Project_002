# Project_002: Disaster Response Pipeline Project
The second Udacity project in the Datascientist Nanodegree

### Table of Contents

1. [Installation](#installation)
2. [How to get started](#start)
3. [File Descriptions](#files)
4. [Motivation](#motivation)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This project relies on the following Packages:
- nltk
- pickle
- re
- pandas
- sklearn
- sqlalchemy

The code should run with no issues using Python versions 3.*.
## How to get started <a name="start"></a>
1. To run ETL pipeline that cleans data and stores in database from the data folder run: 

    `python process_data.py disaster_messages.csv disaster_categories.csv [choose_a_name].db`

2. To run ML pipeline that trains classifier and saves from the models folder run: 

    `python train_classifier.py ../data/DisasterResponse.db [choose_a_name].pkl` 
    
3. from the app folder run: `python run.py` to run the app

4. in your browser go to Go to `http://0.0.0.0:3001/` and enjoy the app 

## File Descriptions <a name="files"></a>

- app
 - template
    - master.html: main page of web app
    - go.html: classification result page of web app
- run.py: Flask file that runs app

- data
    - disaster_categories.csv: data to process 
    - disaster_messages.csv: data to process
    - process_data.py: Script to process the csv files to the cleaned .db file
    - InsertDatabaseName.db: database to save clean data to

- models
    - train_classifier.py: Script to create the trained model 
    - classifier.pkl: saved model (not in this repository because it is too big for github)

## Motivation <a name="motivation"></a>

This Project is about the implementation of a disaster response pipeline in the shape of a web app.
In order to organize which message to your service like a disaster helpdesk should go to which helping entity,
 a machine learning driven backend analyses the messages and determines what the content is about.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

I must give credit to [Figure Eight](https://www.figure-eight.com/) and [Udacity](https://www.udacity.com) for the data, the templates and the guidance.