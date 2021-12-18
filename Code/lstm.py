# -*- coding: utf-8 -*-
"""
# COVID-19 Fake News Detection Using a Long Short-Term Memory Model

One of the models that we are testing for our COVID-19 fake news detector is a Long Short-Term Memory (LSTM) Model. A LSTM model is an excellent choice when working with textual data. It is a form of Recurrent Neural Network (RNN): A RNN works well with sequences such as text, but can suffer from vanishing gradient when sequences become too long. The vanishing gradient problem can prove catastrophic with text, in which the meaning of a word is often informed (at least in part) by the words surrounding it. A LSTM aims to resolve the vanishing gradient problem.

## Import Libraries

We use the following libraries to prepare our data and to build the LSTM model.
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
from ast import literal_eval
import numpy as np
import json

import statistics

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix, accuracy_score,\
                            classification_report, roc_curve, auc

from keras.models import load_model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
# %matplotlib inline
import altair as alt
import seaborn as sns


#set a value to random state to allow for reproducibility with 'random' functions
random_state = 42

"""## Preparing Data for LSTM Model

### Read in Data

We import the final training, validation, and test dataframes that we will use for the COVID-19 fake news detector.
"""

def import_X(file_path):
  '''
  input:  file path of csv file to be read in
  output: X values as list of integer sequences for padding
  '''
  df = pd.read_csv(file_path)
  seq = list(df['sequence'].apply(literal_eval))

  return seq

def import_y(file_path):
  '''
  input:  file path of csv file to be read in
  output: y values as numpy array
  '''
  df = pd.read_csv(file_path)
  np_arr = np.array(df['label'])

  return np_arr

#obtain X values as list of integer sequences (from tokenized text)
X_train_seq = import_X('/Data/X_train.csv')
X_val_seq = import_X('/Data/X_val.csv')
X_test_seq = import_X('/Data/X_test.csv')

#obtain y values as numpy array
y_train = import_y('/Data/y_train.csv')
y_val = import_y('/Data/y_val.csv')
y_test = import_y('/Data/y_test.csv')

"""### Pad Sequences

A LSTM model requires that the input sequences all have the same dimension. We have varying lengths of sequences (from as short as 3 to as long as 4,875), but a histogram of the data shows that most of the data is no longer than 500 words. As such, we use a maximum length of 500 for our sequences; anything shorter than 500 is padded with zeroes at the beginning of the sequence, and anything longer than 500 is truncated at the end of the sequence.

#### Explore Sequence Lengths
"""

X_len = [len(x) for x in X_train_seq] + [len(x) for x in X_val_seq] + [len(x) for x in X_test_seq]

#check statistics regarding sequence length
print('The longest sequence is', max(X_len), 'words.')
print('The shortest sequence is', min(X_len), 'words.')
print('The median sequence length is', statistics.median(X_len), 'words.')

plt.hist(X_len, bins=range(0, max(X_len) + 250, 250))
plt.title('Distribution of Document Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Counts')
plt.show()

"""#### Perform Padding"""

max_words = 500

def pad_seq(seq_list):
  '''
  input:  list of sequences of varying lengths
  output: list of sequences with length of 250
  '''
  padded = pad_sequences(seq_list,
                         maxlen=max_words,
                         padding='pre',
                         truncating='post')
  
  return padded

X_train = pad_seq(X_train_seq)
X_val = pad_seq(X_val_seq)
X_test = pad_seq(X_test_seq)

"""## Create Basic LSTM Model

Our basic LSTM model contains the following layers:

*   **Embedding layer**: Transforms our text (integer-encoded sequences) into word embeddings.

*   **Dropout layer**: Helps to prevent overfitting by randomly setting input units to 0 at the specified rate.

*   **LSTM layer**

*   **Dropout layer**

*   **Dense layer**: Applies the sigmoid activation function to the output from the previous layer to reduce each sample to a single dimension representing the model's prediction, i.e., real or fake news

We use grid search to find the best hyperparameters for the basic LSTM model from the following options:

*   **Dropout rate**: [0.2, 0.3, 0.4]

*   **Output dimension from LSTM layer**: [16, 32, 64]
"""

#static parameters for LSTM model
input_dim = 26709 #size of vocabulary/input to embedding layer
embed = 100       #vector space/output dimension for word embeddings
n_epoch = 10      #number of epochs to run model
n_batch = 32      #number of samples to include in each batch

"""### Grid Search for Hyperparameter Tuning"""

def lstm_model_configs():

  """
  output: all possible configurations of parameters to grid search for optimal
          hyperparameter tuning
  """

	#set parameters to test
  drop = [.2,.3,.4]     #prevents model from overfitting
  lstm_dim = [16,32,64] #output dimension from LSTM layer

	#create configurations from parameters
  configs = list()

  for a in drop:
    for b in lstm_dim:
      cfg = [a, b]
      configs.append(cfg)
 
  return configs

def lstm_model_fit(config):

  """
  input:  one possible configuration of hyperparameters
  output: lstm model fit with the hyperparameter configuration
  """

  #unpack configurations
  drop, lstm_dim = config

  #initiate LSTM model
  model = Sequential()
  model.add(Embedding(input_dim=input_dim,
                      output_dim=embed,
                      input_length=max_words))
  model.add(Dropout(rate=drop))
  model.add(LSTM(units=lstm_dim))
  model.add(Dropout(rate=drop))
  model.add(Dense(units=1,
                  activation='sigmoid'))
  
  #compile LSTM model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics = ['accuracy'])
  
  #fit LSTM model
  model.fit(X_train, y_train,
            validation_data=(X_val,y_val),
            epochs=n_epoch,
            batch_size=n_batch)
  
  return model

def lstm_model_predict(model):

  """
  input:  fit LSTM model
  output: predictions on test data using fit model
  """

	#use model to obtain predictions on test data
  pred = model.predict(X_val)
  
  #convert probabilities to class labels
  labels = (pred > 0.5).astype('int32')

  return labels

def lstm_model_acc(labels):

  """
  input:  class labels from using fit model to obtain predictions
  output: accuracy score of current model/configurations
  """

  return accuracy_score(y_val, labels)

def lstm_grid_search(cfg_list):

  """
  input:  list of LSTM model hyperparameter configurations
  output: dictionary of configuration with applicable accuracy score
  """

  #create list to store accuracy scores
  scores = []

  #obtain accuracy score for each configuration of hyperparameters
  for cfg in cfg_list:
    model = lstm_model_fit(cfg)
    labels = lstm_model_predict(model)
    scores.append(lstm_model_acc(labels))

  return scores

#create list of LSTM model hyperparameter configurations
cfg_list = lstm_model_configs()

#perform grid search to obtain optimal configuration
scores = lstm_grid_search(cfg_list)

#save accuracy scores to file
with open('new_scores2.txt', 'w') as convert_file:
     convert_file.write(json.dumps(scores))

!cp new_scores2.txt /Data

"""### Retrieve Optimal Parameters"""

#read in accuracy scores from grid search
f = open("/Data/new_scores2.txt", "r")
scores = f.read()

#convert string to list
scores = scores.strip('][').split(', ')

#find maximum accuracy and position in list
print('The highest accuracy score is:', max(scores))

#find index of maximum accuracy score
acc_index = scores.index(max(scores))

#retrieve hyperparameters for highest accuracy score
drop, lstm_dim = cfg_list[acc_index]

print('The optimal dropout rate is', drop)
print('The optimal dimension for the LSTM layer is', lstm_dim)

"""Given that our optimal parameters are in the middle of ranges that we tested for each parameter ([0.2, 0.3, 0.4] for dropout rate, and [16, 32, 64] for output dimension from LSTM layer), we are going to stick with those values and not continuing to test them.

### Determine Optimal Batch Size

Now that we have our parameters, we are going to test if using a different batch size will give us better results.
"""

def lstm_batch_fit(batch_size):

  """
  input:  one possible batch size
  output: lstm model fit with the batch size
  """

  #initiate LSTM model
  model = Sequential()
  model.add(Embedding(input_dim=input_dim,
                      output_dim=embed,
                      input_length=max_words))
  model.add(Dropout(rate=drop))
  model.add(LSTM(units=lstm_dim))
  model.add(Dropout(rate=drop))
  model.add(Dense(units=1,
                  activation='sigmoid'))	
  
  #compile LSTM model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics = ['accuracy'])
  
  #fit LSTM model
  model.fit(X_train, y_train,
            validation_data=(X_val,y_val),
            epochs=n_epoch,
            batch_size=batch_size)
  
  return model

batch_sizes = [8,16,32,64]

batch_scores = {}

for size in batch_sizes:
    model = lstm_batch_fit(size)
    labels = lstm_model_predict(model)
    batch_scores[size] = lstm_model_acc(labels)

batch_scores = {8: 0.8016372795969773,
                16: 0.7965994962216625,
                32: 0.7953400503778337,
                64: 0.8110831234256927}

#save batch scores to file
with open('lstm_batch_scores.txt', 'w') as f:
     f.write(json.dumps(batch_scores))

!cp lstm_batch_scores.txt /Data

n_batch = 64

"""Using a batch size of 64 gives us the best scores, so we use that value going forward.

### Determine Optimal Epoch Size
We use early stopping to see if we can improve the accuracy score using a different epoch size. We are testing a larger epoch size of 200 and setting our patience, which is the number of epochs without any improvement before training is stopped, to 30.
"""

#create early stopping callback for model training
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=30)

#create callback to save best model observed during training
mc = ModelCheckpoint('best_lstm_model.h5',
                     monitor='val_loss',
                     save_best_only=True)

n_epoch = 200

#initiate LSTM model
model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=embed, input_length=max_words))
model.add(Dropout(rate=drop))
model.add(LSTM(units=lstm_dim))
model.add(Dropout(rate=drop))
model.add(Dense(units=1, activation='sigmoid'))
  
#compile LSTM model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
  
#fit LSTM model
lstm_hist = model.fit(X_train, y_train, validation_data=(X_val,y_val), 
                      epochs=n_epoch, batch_size=n_batch, 
                      callbacks=[early_stopping,mc])

"""The model achieved the best performance (82.43%) at the fifth epoch."""

n_epoch = 5

"""### Determine Optimal Decision Threshold

With a binary classifier such as the one we are using, Keras sets a default threshold of 0.5 for determining the label of a sample; that is, determining whether it is true or fake. We use a ROC curve to see if we can find an optimal threshold to maximize the performance of our model.
"""

#initiate LSTM model with best parameters
model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=embed, input_length=max_words))
model.add(Dropout(rate=drop))
model.add(LSTM(units=lstm_dim))
model.add(Dropout(rate=drop))
model.add(Dense(units=1, activation='sigmoid'))
  
#compile LSTM model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
#fit LSTM model
model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch)

#obtain predictions on X values in the validation set using fit model
y_pred = model.predict(X_val).ravel()

#calculate false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_val, y_pred)

#calculate geometric mean to find optimal threshold
gmeans = np.sqrt(tpr * (1-fpr))

# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='LSTM')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

#axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

#title
plt.title('ROC Curve for Basic LSTM Model')

#show the plot
plt.show()

auc = auc(fpr, tpr)
print('Area under curve:', auc)

"""## Evaluate Models"""

def plot_metric(history, metric):
    '''
    input:
      history from fit model
      metrics to plot
    output: plot of metric
    '''
    #obtain metrics
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]

    #plot metrics
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])

    plt.show()

def pred_classes(model):
  '''
  input:  trained model
  output: predicted classes running model on testing data
  '''
  #use model to predict y values
  y_pred = model.predict(X_test)

  #convert probabilities to class labels
  classes = (y_pred > thresholds[ix]).astype('int32')

  return classes

def conf_matrix(classes):
  '''
  input:  predicted classes
  output: confusion matrix using predicted classes
  '''
  #create confusion matrix
  cm = confusion_matrix(y_test, classes)
  sns.heatmap(cm, annot=True, fmt='d')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()

"""### Basic LSTM Model with Optimal Hyperparameters and Decision Threshold

Now that we have tuned our model, it is time to test its performance on our test dataset.
"""

#initiate LSTM model with best parameters
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=input_dim, output_dim=embed, input_length=max_words))
lstm_model.add(Dropout(rate=drop))
lstm_model.add(LSTM(units=lstm_dim))
lstm_model.add(Dropout(rate=drop))
lstm_model.add(Dense(units=1, activation='sigmoid'))
  
#compile LSTM model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
#fit LSTM model
lstm_model_hist = lstm_model.fit(X_train, y_train, 
                                 validation_data=(X_test,y_test),
                                 epochs=n_epoch, batch_size=n_batch)

plot_metric(lstm_model_hist, 'loss')

plot_metric(lstm_model_hist, 'accuracy')

lstm_classes = pred_classes(lstm_model)

conf_matrix(lstm_classes)

print(classification_report(y_test,lstm_classes))

"""### Bidirectional LSTM Model

We modify our model by using a bidirectional LSTM layer instead of a LSTM layer and evaluate performance.
"""

#initiate bidirectional LSTM model with best parameters
bi_lstm_model = Sequential()
bi_lstm_model.add(Embedding(input_dim=input_dim, output_dim=embed, input_length=max_words))
bi_lstm_model.add(Dropout(rate=drop))
bi_lstm_model.add(Bidirectional(LSTM(units=lstm_dim)))
bi_lstm_model.add(Dropout(rate=drop))
bi_lstm_model.add(Dense(units=1, activation='sigmoid'))
  
#compile LSTM model
bi_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
#fit LSTM model
bi_lstm_model_hist = bi_lstm_model.fit(X_train, y_train, 
                                       validation_data=(X_test,y_test),
                                       epochs=n_epoch, batch_size=n_batch)

plot_metric(bi_lstm_model_hist, 'loss')

plot_metric(bi_lstm_model_hist, 'accuracy')

bi_lstm_classes = pred_classes(bi_lstm_model)

conf_matrix(bi_lstm_classes)

print(classification_report(y_test,bi_lstm_classes))

"""### LSTM Model with Additional Dense Layer

We check performance after adding an additional dense layer before the final output.
"""

#initiate modified LSTM model with best parameters
lstm_model2 = Sequential()
lstm_model2.add(Embedding(input_dim=input_dim, output_dim=embed, input_length=max_words))
lstm_model2.add(Dropout(rate=drop))
lstm_model2.add(LSTM(units=lstm_dim))
lstm_model2.add(Dropout(rate=drop))
lstm_model2.add(Dense(units=lstm_dim, activation='relu'))
lstm_model2.add(Dropout(rate=drop))
lstm_model2.add(Dense(units=1, activation='sigmoid'))
  
#compile LSTM model
lstm_model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
#fit LSTM model
lstm_model2_hist = lstm_model2.fit(X_train, y_train, 
                                  validation_data=(X_test,y_test),
                                  epochs=n_epoch, batch_size=n_batch)

plot_metric(lstm_model2_hist, 'loss')

plot_metric(lstm_model2_hist, 'accuracy')

lstm2_classes = pred_classes(lstm_model2)

conf_matrix(lstm2_classes)

print(classification_report(y_test,lstm2_classes))
