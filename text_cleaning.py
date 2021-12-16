"""
# Text Cleaning

The purpose of this notebook is to preprocess and clean our textual data as well as to break it into training, validation, and test sets to prepare for running our various COVID-19 fake news detection classifiers.

## Import Libraries

We use the following libraries to clean our data and prepare it for training and testing our models.
"""

import pandas as pd
import numpy as np
import json

import string
import nltk
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split


#set a value to random state to allow for reproducibility with 'random' functions
random_state = 42

"""## Read in Data

We read in our final dataset on which we will perform our text cleaning.
"""

#read in final dataset
data = pd.read_csv('/Data/FINAL_with_articles_CSV.csv')

#shuffle rows to better distribute 0 and 1 labels throughout dataframe
data = data.sample(frac=1,random_state=random_state).reset_index(drop=True)

#check shape of dataframe
data.shape

#check if null values
data[data['text'].isnull()]

#remove null row
data = data.drop(labels=[4055], axis=0).reset_index(drop=True)

data.isnull().any()

#preview data
data.head()

"""## Clean Data

We clean our text before running our various classifiers. Throughout this process, we have tried various degrees of text cleaning to find what works best. Given the fact that our final dataset contains some documents that are significantly long, we attempt to remove as many unnecessary characters and tokens as possible, as well as to reduce the dimensionality of the data, in particular for running our LSTM model.
"""

#suppress warnings
pd.set_option('mode.chained_assignment',None)

#download stopwords
nltk.download('stopwords')

"""### Basic Text Cleaning

We perform basic text cleaning by making the text lowercase and removing punctuation, numerical digits, and unicode characters.
"""

def basic_cleaning(text):
  '''
  input: line of text from dataframe
  output: lowercase text with removal of unnecessary characters
  '''
  #make text lower case
  new_text = text.lower()

  #remove punctuation
  new_text = new_text.translate(str.maketrans('', '', string.punctuation))

  #remove numeric digits
  new_text = new_text.translate(str.maketrans('', '', string.digits))
  
  #remove unicode characters
  new_text = new_text.encode("ascii", "ignore")
  new_text = new_text.decode()

  return new_text

#apply basic_cleaning() function to text
data['clean'] = data['text'].apply(basic_cleaning)

#preview data
data.head()

"""### Stopword Removal

We remove stop words because they will not add much value to our analysis.
"""

def stop_word_removal(text):
  '''
  input: line of text (after basic cleaning) from dataframe
  output: text with stop words removed
  '''
  #split text into list of words
  words = text.split()

  #remove stopwords
  words = [word for word in words if word not in stopwords.words('english')]

  #join text back together
  cleaned = ' '.join(words)

  return cleaned

#apply stop_word_removal() function to text
data['clean'] = data['clean'].apply(stop_word_removal)

#preview data
data.head()

"""### Rare Word Removal

We remove rare words, which we define as those that appear only once in the entire corpus. We do this to limit our corpus somewhat, in particular those words that will not be very informative to our models.
"""

def get_rare_words(df,col):
  '''
  input: dataframe and col to analyze
  output: list of words that occur once in column (rare words)
  '''
  #create a list of words (tokens) from our column
  tokens = ' '.join(df[col]).split()

  #obtain dictionary with word as key and occurrence as value
  freq_dist = dict(nltk.FreqDist(tokens))

  #create list of all words that only appear once in column
  rare_words = [k for k,v in freq_dist.items() if v == 1]

  return rare_words

#obtain rare words (occur only once in dataframe)
rare_words = get_rare_words(data,'clean')

def rare_word_removal(text):
  '''
  input: line of text (after basic cleaning and stop word removal) from dataframe
  output: text with rare words removed
  '''
  #split text into list of words
  words = text.split()

  #remove rare words
  words = [word for word in words if word not in rare_words]

  #join text back together
  cleaned = ' '.join(words)

  return cleaned

#apply rare_word_removal() function to text
data['clean'] = data['clean'].apply(rare_word_removal)

#preview data
data.head()

"""### Stem Text

We stem the text to limit our corpus and ensure that words with the same stems (and thus similar meanings) are tokenized using the same integer.
"""

#initiate stemmer
snowball = nltk.stem.SnowballStemmer(language='english')

def stem_text(text):
  '''
  input: line of text (after basic cleaning/stop and rare word removal) from dataframe
  output: stemmed text
  '''
  #split text into list of words
  words = text.split()

  #stem text
  stemmed = [snowball.stem(word) for word in words]

  #join text back together
  cleaned = ' '.join(stemmed)

  return cleaned

#apply stem_text() function to text
data['stem'] = data['clean'].apply(stem_text)

#preview data
data.head()

"""### Remove Empty Rows

Finally, before proceeding with tokenizing the data, we remove any rows from our dataframe where the length of the 'stem' column is less than 3 so as to remove those rows that add little or no value to the data.
"""

data['length'] = data['stem'].apply(lambda row:len(row.split()))

#preview data
data.head()

final_data = data[data['length']>2].reset_index(drop=True)

#preview data
final_data.head()

#save final, cleaned dataframe to Google Drive
final_data.to_csv('cleaned_df.csv')
!cp cleaned_df.csv /Data

"""## Tokenize Data

In order to prepare the text to feed into our models, we tokenize it using the *texts_to_sequences* function in the Tokenizer class. This function cleans the data, splits it into words, and assigns an integer representation to each word. From there, a line of text is represented as a sequence of integers.
"""

#initiate and fit tokenizer
tokenizer = Tokenizer(oov_token='oov')
tokenizer.fit_on_texts(final_data['stem'])

#turn text into sequences of integers
sequences = tokenizer.texts_to_sequences(final_data['stem'])

#create index
vocab = tokenizer.word_index

#check length of vocabulary created from tokenizer
print('The length of the vocabulary is', len(vocab), 'words.')

#save vocab to file
with open('vocab_index.txt', 'w') as f:
     f.write(json.dumps(vocab))

!cp vocab_index.txt /Data

"""## Create Training, Validation, and Test Datasets

Finally, we split the data into training, validation, and test datasets using a 70/15/15 split.
"""

token_df = pd.DataFrame(data={'sequence':sequences,
                              'label':final_data['label']})

token_df.head()

#70/30 split between training and validation/test data
X_train, X_val_test, y_train, y_val_test = train_test_split(token_df['sequence'], token_df['label'],
                                                            test_size=0.3,
                                                            stratify=token_df['label'],
                                                            random_state=random_state)

#50/50 split between validation and test data
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test,
                                                test_size=0.5,
                                                stratify=y_val_test,
                                                random_state=random_state)

"""## Save Training, Validation, and Test Datasets"""

X_train.to_csv('X_train.csv')
!cp 'X_train.csv' /Data

y_train.to_csv('y_train.csv')
!cp 'y_train.csv' /Data

X_val.to_csv('X_val.csv')
!cp 'X_val.csv' /content/drive/MyDrive/saids697/Data

y_val.to_csv('y_val.csv')
!cp 'y_val.csv' /Data

X_test.to_csv('X_test.csv')
!cp 'X_test.csv' /Data

y_test.to_csv('y_test.csv')
!cp 'y_test.csv' /Data
