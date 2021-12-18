
import pandas as pd
import re

from google.colab import drive
drive.mount('/content/drive')

"""# **Pulling Data**
Our data comes from different sources following cells are combining them in one data frame

# Following is twitter data - column Text is in fact content
"""

## https://ieee-dataport.org/open-access/covid-19-fake-news-infodemic-research-dataset-covid19-fnir-dataset
fake_news = pd.read_csv("/content/drive/MyDrive/saids697/Data/fakeNews.csv")#[['Text',"Binary Label"]].drop_duplicates()
fake_news = fake_news.rename(columns={"Binary Label": "label", "Text":'text'}) 
fake_news = fake_news[['text','label']]
fake_news

"""# Following is NewsAPI data - column Text is in fact content"""

## Original Scraped Data 
combined_trues1 = pd.read_csv('/content/drive/MyDrive/saids697/NEWSAPI1129_US_628_828/combined_csv.csv')[['content']].drop_duplicates()
combined_trues1 = combined_trues1.rename(columns={"content":'text'}) 
combined_trues1['label'] = 1 #we have to add a labeled column 1 since all are true

## Original Scraped Data 
combined_trues2 = pd.read_csv('/content/drive/MyDrive/saids697/NEWSAPI1116_US_82820/combined_csv.csv')[['content']].drop_duplicates()
combined_trues2 = combined_trues2.rename(columns={"content":'text'}) 
combined_trues2['label'] = 1 #we have to add a labeled column 1 since all are true

## Original Scraped Data 
combined_trues3 = pd.read_csv('/content/drive/MyDrive/saids697/NEWSAPI1129_428/combined_csv.csv')[['content']].drop_duplicates()
combined_trues3 = combined_trues3.rename(columns={"content":'text'}) 
combined_trues3['label'] = 1 #we have to add a labeled column 1 since all are true

## Original Scraped Data 
combined_trues4 = pd.read_csv('/content/drive/MyDrive/saids697/NEWSAPI1129_228/combined_csv.csv')[['content']].drop_duplicates()
combined_trues4 = combined_trues4.rename(columns={"content":'text'}) 
combined_trues4['label'] = 1 #we have to add a labeled column 1 since all are true

combined_trues4

"""#This data set contains both misinfo and non misinfo"""

## https://data.mendeley.com/datasets/zwfdmp5syg/1/files/3063167e-1d3b-4604-a630-16016a84e8db
fake_new = pd.read_excel('/content/drive/MyDrive/saids697/Data/fake_new_dataset.xlsx')[["text","label"]].drop_duplicates()
fake_new

"""#Princeton Data
While most of data is in fact about misinformation, it has to be infosized that it is ABOUT it, fact checkinga nd mythbasting articles - added to teh dataset for the sanity check as a lot of aarticles debunking misinformation also contains 'juicy' key words
"""

## https://esoc.princeton.edu/publications/esoc-covid-19-misinformation-dataset 
covid_misinfo = pd.read_excel('/content/drive/MyDrive/saids697/Data/Covid_Misinfo.xlsx').drop_duplicates()
covid_misinfo = covid_misinfo[covid_misinfo['Spanish']=='English']
frames = [covid_misinfo[covid_misinfo['fake']==0], covid_misinfo[covid_misinfo['fake']==1]]
covid_misinfo = pd.concat(frames) 
covid_misinfo['label'] = covid_misinfo['fake'].apply(lambda x: (x-1)**2)
covid_misinfo = covid_misinfo[['Title','label']]
covid_misinfo.label.value_counts()

pd.set_option('display.max_colwidth', None)

covid_misinfo = covid_misinfo.rename(columns={ "Title":'text'})

"""#Infowars labeled manually

"""

from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())

worksheet = gc.open('Infowars').sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()
print(rows)

# Convert to a DataFrame and render.
import pandas as pd
iw1 = pd.DataFrame.from_records(rows)

iw1 = iw1[1:]
iw1.columns = ['url','text','title','label']
iw1['label'] = 0
iw1 = iw1[['text','label']]



all_news_df = pd.concat([fake_news,fake_new,combined_trues1,combined_trues2,combined_trues3, combined_trues4, covid_misinfo, iw1]).dropna()

print("Number of Fake News Entries: ", len(all_news_df[all_news_df['label']==0]))
print("Number of True News Entries: ", len(all_news_df[all_news_df['label']==1]))

print("Number of Fake News Entries: ", len(all_news_df[all_news_df['label']==0].drop_duplicates()))
print("Number of True News Entries: ", len(all_news_df[all_news_df['label']==1].drop_duplicates()))

#all_news_df.to_csv('/content/drive/MyDrive/saids697/Data/FINAL_CSV.csv', index = False, header=True)

df = all_news_df.drop_duplicates()

all_news_df.to_csv()

"""# **Data preprocessing**"""

df['category'] = df['label']
df['category'] = df.category.replace({1:'news', 0:'misleading'})

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.15, 
                                                  random_state=17, 
                                                  stratify=df.label.values)

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

"""#Bert"""



df.groupby(['category', 'label', 'data_type']).count()

!pip install transformers

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)

!pip install torch

import torch



"""TODO : Experiment with max length"""

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

len(dataset_train)

len(dataset_val)



"""## Setting up BERT Pretrained Model"""

from transformers import BertForSequenceClassification
label_dict = {'news':1, 'misleading':0}
inv_label_dict = {'news':1, 'misleading':0}
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False,
                                                      hidden_dropout_prob = 0.11)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)

epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')



def acc_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

import torch
import pandas as pd
from tqdm.notebook import tqdm

for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_acc = acc_score_func(predictions, true_vals)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Acc Score : {val_acc}')
    tqdm.write(f'f1 Score : {val_f1}')

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load('Models/_model.model', map_location=torch.device('cpu')))

