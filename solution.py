import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import deepmatcher as dm
import recordlinkage as rl
import xarray
from IPython.display import display
from os.path import join
from sklearn.utils import shuffle
import py_entitymatching as em
import nltk
nltk.download('punkt')

def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = ["left_" + col for col in tpls_l.columns]
    tpls_r.columns = ["right_" + col for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR


# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
labels = pd.read_csv(join('data', "train.csv"))


# 2 Preprocess data 

# Rename columns
ltable.columns = ['left_id', 'left_title', 'left_category', 
    'left_brand', 'left_modelno', 'left_price']  
rtable.columns = ['right_id', 'right_title', 'right_category', 
    'right_brand', 'right_modelno', 'right_price']
labels.columns = ['left_id', 'right_id', 'label']

# add id column to label
labels.insert(0, 'id', range(0, len(labels)))

# merge data into labels
labels = labels.merge(ltable, how = 'left', on='left_id')
labels = labels.merge(rtable, how = 'left', on='right_id')

# split data into train, validation, and test (optional)
dm.data.split(labels, '/content/drive/MyDrive/CS4400_FinalProject', 
    'train.csv', 'validation.csv', 'test.csv', split_ratio = [0.7499, 0.001, 0.25])

# load and process labeled training, validation and test CSV data
train, valid = dm.data.process(
    path='/content/drive/MyDrive/CS4400_FinalProject',
    train='train.csv',
    validation = 'validation.csv',
    ignore_columns = ['left_id', 'right_id'])


# 3. Define Model

model = dm.MatchingModel()

# 4. Train Model
model.run_train(train, valid, pos_neg_ratio = 8.65, epochs = 15,
    best_save_path = 'best_model.pth')
# 5. Evaluate Model

# Load Model if not training
# model.load_state('/content/drive/MyDrive/CS4400_FinalProject/best_model.pth')

# Evaluate Model (if desired)
# model.run_eval(test)

# Get original training data to exlcude
train0 = pd.read_csv(join('data', "train.csv"))
train0.columns = ['left_id', 'right_id', 'label']

# Rename columns
ltable.columns = ['id', 'title', 'category', 
    'brand', 'modelno', 'price']  
rtable.columns = ['id', 'title', 'category', 
    'brand', 'modelno', 'price']  

# Perform Blocking
A = em.read_csv_metadata(
    '/content/drive/MyDrive/CS4400_FinalProject/data/ltable.csv', key='id')
B = em.read_csv_metadata(
    '/content/drive/MyDrive/CS4400_FinalProject/data/rtable.csv', key='id')
ob = em.OverlapBlocker()
candset_df = ob.block_tables(A, B, 'title', 'brand', overlap_size=1, 
    l_output_attrs=['id', 'title', 'category', 'brand', 'modelno', 'price'], 
    r_output_attrs=['id', 'title', 'category', 'brand', 'modelno', 'price'],
    l_output_prefix = 'left_', r_output_prefix = 'right_', allow_missing = True,
    n_jobs = 2)

candset_df.columns.values[0] = 'id'

# Get unlabeled.csv
candset_df.to_csv('unlabeled.csv')

# Process Unlabeled
unlabeled = dm.data.process_unlabeled(
    path='/content/drive/MyDrive/CS4400_FinalProject/unlabeled.csv',
    trained_model = model)

# Get predicitions
predictions = model.run_prediction(unlabeled)
predictions.to_csv('predictions0.csv')
predictions = pd.read_csv("predictions0.csv")
training_pairs = list(map(tuple, train0[["left_id", "right_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
predictions["id"] = predictions["id"].astype(int)
candset_df["id"] = candset_df["id"].astype(int)
predictions = predictions.merge(candset_df, how = "left", on = "id")

# Find matching pairs
matching_pairs = predictions.loc[predictions.match_score.values >= 0.5, ["left_id", "right_id"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[train0.label.values == 1, ["left_id", "right_id"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

# Exclude original training data 
pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)

