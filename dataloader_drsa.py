import os
import numpy as np
from torch.utils.data import DataLoader
import random
import torch
from sksurv.datasets import load_flchain
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# please note that, s[1] is event time, s[2] is observation time
# The event time and observation time in these codes are the right ones
# The event time and observation time in the original data are the wrong ones

# class SurvivalDataSet_Serum(INPUT_FILE="DRSA-PyTorch/DRSA-hua/data_with_summaries_and_embeddings.csv"):
#     def __init__(self, is_partial=True, INPUT_FILE="DRSA-PyTorch/DRSA-hua/data_with_summaries_and_embeddings.csv"):
#         self.is_partial = is_partial
        
#         self.data = []
#         self.labels = []
#         self.event_times = []
#         self.observation_times = []
#         self.win = []

#         data_1 = load_flchain()
#         data_2 = pd.read_csv(INPUT_FILE)

#         x = data_2.loc[:2000,['age','chapter','creatinine','flc.grp','kappa','lambda','mgus','sex']].copy()
#         generated_texts = data_2.loc[:2000,['generated_texts']].copy()
#         embeddings = data_2.loc[:2000,['embeddings']].copy()
#         y = data_1[1][:2000].copy()

#         # Create a copy of the data
#         x_normalized = x.copy()

#         # 1. Numerical data (age, creatinine, kappa, lambda)
#         numerical_cols = ['age', 'creatinine', 'kappa', 'lambda', 'flc.grp']
#         scaler = MinMaxScaler()
#         x_normalized[numerical_cols] = scaler.fit_transform(x[numerical_cols])
#         # 2. Categorical data (chapter, mgus, sex)
#         categorical_cols = ['chapter', 'mgus', 'sex']
#         for col in categorical_cols:
#             # Convert to numeric first
#             le = LabelEncoder()
#             encoded_values = le.fit_transform(x[col])
#             # Then normalize to [0,1]
#             x_normalized[col] = encoded_values / (len(le.classes_) - 1)
        
      
#         event_time = int(s[1]) # the data in the second column
#         observation_time = int(s[2]) # the data in the third column
#         if self.win:
#             if observation_time >= event_time:  # uncensored
#                 self.data.append(t_indices)  # data only conclude indices use this to get embedding
#                 self.event_times.append(event_time / discount)
#                 self.observation_times.append(observation_time / discount)
#                 self.labels.append([0., 1.])  # we win means we dead
#                 self.win.append(1)
#         else:  # censored
#             if observation_time < event_time:  # uncensored
#                 self.data.append(t_indices)  # data only conclude indices use this to get embedding
#                 self.event_times.append(event_time / discount)
#                 self.observation_times.append(observation_time / discount)
#                 self.labels.append([1., 0.])  # so far we always lose, it means we still survial
#                 self.win.append(0)
#         self.max_d = max_d

#         fi.close()
#         self.feature_size = len(self.data[0])
#         self.data = np.array(self.data)
#         self.labels = np.array(self.labels)
#         self.event_times = np.array(self.event_times)
#         self.observation_times = np.array(self.observation_times)
#         self.win = np.array(self.win)

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return {
#             'x': self.data[idx],
#             'label': self.labels[idx],
#             'event_time': self.event_times[idx],
#             'observation_time': self.observation_times[idx],
#             'win': self.win[idx]
#         }

# class SurvivalDataSet_Win():
#     def __init__(self, INPUT_FILE, win, discount=1):
#         self.discount = discount
#         self.data = []
#         self.labels = []
#         self.event_times = []
#         self.observation_times = []
#         self.win = []
#         self.is_win = win

#         fi = open(INPUT_FILE, 'r')
#         COUNT = 1
#         max_d = -1
#         for line in fi:
#             s = line.split(' ')
#             slen = len(s)
#             t_indices = []

#             for i in range(3, slen):
#                 w = s[i].split(':')
#                 td = int(w[0])
#                 t_indices.append(td)
#                 max_d = max(td, max_d)
#             event_time = int(s[1]) # the data in the second column
#             observation_time = int(s[2]) # the data in the third column
#             if self.win:
#                 if observation_time >= event_time:  # uncensored
#                     self.data.append(t_indices)  # data only conclude indices use this to get embedding
#                     self.event_times.append(event_time / discount)
#                     self.observation_times.append(observation_time / discount)
#                     self.labels.append([0., 1.])  # we win means we dead
#                     self.win.append(1)
#             else:  # censored
#                 if observation_time < event_time:  # uncensored
#                     self.data.append(t_indices)  # data only conclude indices use this to get embedding
#                     self.event_times.append(event_time / discount)
#                     self.observation_times.append(observation_time / discount)
#                     self.labels.append([1., 0.])  # so far we always lose, it means we still survial
#                     self.win.append(0)
#         self.max_d = max_d

#         fi.close()
#         self.feature_size = len(self.data[0])
#         self.data = np.array(self.data)
#         self.labels = np.array(self.labels)
#         self.event_times = np.array(self.event_times)
#         self.observation_times = np.array(self.observation_times)
#         self.win = np.array(self.win)

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return {
#             'x': self.data[idx],
#             'label': self.labels[idx],
#             'event_time': self.event_times[idx],
#             'observation_time': self.observation_times[idx],
#             'win': self.win[idx]
#         }
        
class SurvivalDataSet():
    def __init__(self, INPUT_FILE, config, is_train=True):
        self.is_train = is_train
        self.data = []
        self.events = []
        self.times = []
        
        if self.use_embedding:
            self.embeddings = []
        self.feature_size = 0
        self.discount_factor = config['discount_factor']

        fi = open(INPUT_FILE, 'r')
        COUNT = 1
        max_d = -1
        self.finish_epoch = False
        for line in fi:
            s = line.split(' ')
            slen = len(s)
            t_indices = []
            for i in range(3, slen):
                w = s[i].split(':')
                td = int(w[0])
                t_indices.append(td)
                max_d = max(td, max_d)
            event_time = int(s[1]) # the data in the second column
            observation_time = int(s[2]) # the data in the third column
            if observation_time >= event_time:  # uncensored
                self.data.append(t_indices)  # data only conclude indices use this to get embedding
                self.event_times.append(event_time / discount)
                self.observation_times.append(observation_time / discount)
                self.labels.append([0., 1.])  # we win means we dead
                self.win.append(1)
            else:  # censored
                self.data.append(t_indices)  # data only conclude indices use this to get embedding
                self.event_times.append(event_time / discount)
                self.observation_times.append(observation_time / discount)
                self.labels.append([1., 0.])  # so far we always lose, it means we still survial
                self.win.append(0)
        self.max_d = max_d
        fi.close()
        self.feature_size = len(self.data[0])
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.event_times = np.array(self.event_times)
        self.observation_times = np.array(self.observation_times)
        self.win = np.array(self.win)
        # print("data size ", self.size, "\n")

    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        # return self.data[idx], self.labels[idx], self.event_times[idx], self.observation_times[idx], self.win[idx]
        return {
            'x': self.data[idx],
            'label': self.labels[idx],
            'event_time': self.event_times[idx],
            'observation_time': self.observation_times[idx],
            'win': self.win[idx]
        }
    
class DataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            # When we reach the end, create a new iterator
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
    
if __name__ == "__main__":
    dataset = SurvivalDataSet("data/CLINIC/train.yzbx.txt")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    # data_win = dataloader.dataset.data_win
    print(len(dataset))

    # print(dataset[0])

    # print(len(dataset[0]))

    # fi = open("data/CLINIC/train.yzbx.txt", 'r')
    # count = 0
    # for line in fi:
    #     count += 1
    # print(count)
    # fi.close()

