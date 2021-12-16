# This README is under construction until 12/19/2021

# COVID-19-Misinformation-Classificaton-System

Since the discovery of the novel corornavirus-2019, there has been a lot of misinformation surrounding it. This includes the origin of the virus, the fatalness of it, the vaccine efficacy, and most recently the necessity of COVID-19 booster shot. Much of the misinformation surrounding the vaccine is extremely harmful to groups such as those of Chinese/Asian descent, medical proffesionals, WHO and CDC employees, those who are immunocompromised, and many more. Trialing many different machine learning models, we aim to create a useful classifer to predict whether a news article - given just the title or full text - is reliable or not (True or Fake). 

### PLEASE NOTE: The models as they are currently published are for educational purposed only and should not to be used as a fact-checking substitute by public and/or private organizations. 

## The Data 
 The data used for this analysis is a mixure of previously coded datasets, as well as newly encoded datasets by our team. 
 The code we gathered from existing sources can be found at the following sites:
 
  https://ieee-dataport.org/open-access/covid-19-fake-news-infodemic-research-dataset-covid19-fnir-dataset
  
  https://data.mendeley.com/datasets/zwfdmp5syg/1 
  
  https://esoc.princeton.edu/publications/esoc-covid-19-misinformation-dataset

All data sets, along with the final, combined data set can be found in our GitHub repository, along with all code used for cleaning.

The final dataset used for analysis is the "cleaned_df.csv" found in the repo files

## The Models 
  Please be sure to comment out the GridSearchCV portions of the code, and they will significantly slow down execution time. 
  
## Dummy Classifiers

## Logistic Regression
  ### tf-idf
  ### Bag of Words
  ### Ngram(1,2)
  ### Ngram(1,3)
  ### Gensim Word2Vec
## Random Forest
  ### Ngram(1,3)
## Emsemble Model
## Long Short Term Memory (LSTM)
## Bidirectional Encoder Representations from Transformers (BERT)
