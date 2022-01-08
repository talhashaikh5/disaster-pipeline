
# Disaster Response Pipeline

## Project Description
In this project, we build a machine learning model to identify messages into multiple classes, messages from differnt sources where collected dusring a disaster and labled in differnt fields like medical_issue, rescue etc. Agenda in this project is to build a model using this data which will help in future to identify what help messages and communiate them to differnt aiding departments.
We used multioutput classifier to classify messages. randomforestclassifer is used to estemator. After building a classifier we made a web app using flask where we can input a message and get classification results.


![Screenshot of Web App]()

## File Description
~~~~~~~
disaster-response
├── .gitattributes
├── .gitignore
├── app/
│   ├── run.py
│   └── templates/
│       ├── go.html
│       └── master.html
├── data/
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   ├── ETL Pipeline Preparation.ipynb
│   └── process_data.py
├── LICENSE
├── models/
│   └── train_classifier.py
├── notebooks/
│   ├── .ipynb_checkpoints/
│   │   ├── ETL Pipeline Preparation-checkpoint.ipynb
│   │   ├── ML Pipeline Preparation-checkpoint.ipynb
│   │   └── Untitled-checkpoint.ipynb
│   ├── classifier
│   ├── disaster.db
│   ├── disaster1.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── ETL Pipeline Preparation.ipynb
│   ├── ML Pipeline Preparation.ipynb
│   └── Untitled.ipynb
├── README.md
├── requirement.txt
~~~~~~~
## Installation
1. Install Requirements
    `pip install -r requirement.txt`

2.  To run ETL pipeline that cleans data and stores in database
	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

3. To run ML pipeline that trains classifier and saves
	`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Run the following command in the app's directory to run your web app.
	`python run.py`
	

## File Descriptions
1. App folder including the templates folder and "run.py" for the web application
2. Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. Models folder including  "train_classifier.py" for the Machine Learning model.
4. README file
5. Notebook folder containing multiple different files, which were used for the project building. (Please note: this folder is not necessary for this project to run.)
6. images contains screenshots of project

## Licensing, Authors, Acknowledgements
Many thanks to Figure-8 for making this available to Udacity for training purposes. Special thanks to udacity for the training. Feel free to utilize the contents of this while citing me, udacity, and/or figure-8 accordingly.

### Note: this project doesnot contain moel file you need to generate model file by running 3rd step in installation part