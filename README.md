**Amazon Review Score Level Prediction**
**Project Description**

This project focuses on predicting the score category of Amazon product reviews—low, medium, or high—by analyzing both review text and associated metadata. The dataset, obtained from Kaggle, includes over 500,000 labeled reviews for training and 5,000 unlabeled reviews for testing. A machine learning–based classification approach was applied, resulting in a model that achieved 84% accuracy on the test data.

**Table of Contents**
Overview
,
Dataset
,
Installation
,
Usage
Model Performance
Contributing
License

**Overview**

The objective of this project is to classify Amazon reviews into three score levels: low, medium, and high. The solution incorporates data preprocessing, feature extraction from text, and supervised machine learning techniques to learn patterns from review metadata and textual content. The trained model demonstrates strong predictive performance, achieving 84% accuracy on unseen data.

**Dataset**

The dataset is sourced from Kaggle and contains the following files:

foods_training.csv – 518,358 reviews with labeled score_level used for model training

foods_testing.csv – 5,000 reviews without labels, used for prediction

sample_submission.csv – A sample submission format for Kaggle evaluation

**Dataset Features**

ID – Unique review identifier

productID – Product identifier

userId – User identifier

helpfulness – Helpfulness score of the review

summary – Short review summary

text – Full review content

score_level – Review rating category (low, medium, high)

**Installation**

Install all required dependencies using:

pip install -r requirements.txt

Usage

Prepare the data

Download the dataset from Kaggle

Place the CSV files inside the data/ directory

Train the model

python train_model.py


Generate predictions

python predict.py --input data/foods_testing.csv --output submission.csv


Submit results

Upload submission.csv to Kaggle for evaluation

**Model Performance**

The final classification model achieved an accuracy of 84% in predicting Amazon review score levels, demonstrating effective learning from both textual and metadata features.
