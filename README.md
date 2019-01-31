# Taxi-Trip-Time-Prediction-Competition
It is a Kaggle machine learning competition about predicting taxi trip time in Portugal

This package contains three files and one folder:
- Kaggle_pipeline.py: The python pipeline script to train separate models for A, B, C type data and generate prediction file called "submission.csv"
- df1.csv: The feature file of training data
- df3.csv: The feature file of test data
- all: the folder contains all data (./all/train.csv ./all/sampleSubmission.csv ./all/test_public.csv ./all/metaData_taxistandsID_name_GPSlocation.csv); For uploading, I don't put these four files into this folder. You should do that before running this program.

The output file:
- submission.csv: a file that can be directly submit on Kaggle to evaluate

Execute command:
python Kaggle_pipeline.py ./all/train.csv ./all/sampleSubmission.csv ./all/test_public.csv ./all/metaData_taxistandsID_name_GPSlocation.csv


Execution time:
About 57 minutes (existing training and test features)

Desired Performance:
Public score: 486.86287
Private score: 569.18524
