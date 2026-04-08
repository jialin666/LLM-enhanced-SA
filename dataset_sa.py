import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import ast
from utils import get_best_device
from operator import itemgetter

class SurvivalDataSet(Dataset):
    def __init__(self, csv_path, config):
        # Set up device
        self.device = get_best_device()
        
        self.use_embedding = config['use_embedding_input']
        self.data = []
        self.events = []
        self.times = []

        
        if self.use_embedding:
            self.embeddings = []
        self.feature_size = 0
        self.discount_factor = config['discount_factor']
        
        # print(f"Loading data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        # print(f"Total rows in DataFrame: {len(self.df)}")
        
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            # if the event_time is larger than 4500, then skip the row
            # if float(row['event_time']) > 4000:
            #     continue
            if self.use_embedding:
                if isinstance(row['embeddings'], str):
                    try:
                        # Try different methods to parse the embedding string
                        try:
                            embedding = json.loads(row['embeddings'])
                        except json.JSONDecodeError:
                            embedding = ast.literal_eval(row['embeddings'])
                        self.embeddings.append(embedding)
                    except Exception as e:
                        print(f"Error parsing embedding at row {i}: {str(e)}")
                        print(f"Problematic embedding value: {row['embeddings'][:100]}...")
                        continue
                else:
                    if pd.isna(row['embeddings']):
                        print(f"Warning: NaN embedding at row {i}")
                        continue
                    self.embeddings.append(row['embeddings'])
            
            # # Apply discount factor to time values
            # if row['event_occurred']:
            #     temp_time = float(row['event_time']) / self.discount_factor
            # else:
            #     temp_time = float(row['observation_time']) / self.discount_factor
            self.events.append(row['event'])
            self.times.append(float(row['time']))
            
            # Convert features to appropriate types
            features = []
            for col in config['feature_names']:
                features.append(float(row[col]))
            self.data.append(features)

        # Convert everything to numpy arrays first
        self.data = np.array(self.data, dtype=np.float32)
        self.events = np.array(self.events, dtype=np.int32) 
        self.times = np.array(self.times, dtype=np.float32)
        if self.use_embedding:
            self.embeddings = np.array(self.embeddings, dtype=np.float32)
        
        self.feature_size = self.data.shape[1]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # try:
        if self.use_embedding:
            sample = {
                'x': torch.tensor(self.data[idx], dtype=torch.float32, device=self.device),
                'event': torch.tensor(self.events[idx], dtype=torch.int32, device=self.device),
                'time': torch.tensor(self.times[idx], dtype=torch.float32, device=self.device),
                'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float32, device=self.device)
            }
        else:
            sample = {
                'x': torch.tensor(self.data[idx], dtype=torch.float32, device=self.device),
                'event': torch.tensor(self.events[idx], dtype=torch.int32, device=self.device),
                'time': torch.tensor(self.times[idx], dtype=torch.float32, device=self.device),
            }
        
        return sample
    

class SurvivalDataSet_2021Transformer(Dataset):

    def __init__(self, csv_path, config, is_train=True):
        self.device = get_best_device()
        
        df = pd.read_csv(csv_path)

        # increase column time: if event is 1, then time is event_time, otherwise time is observation_time
        # df['time'] = df.apply(lambda row: row['event_time'] if row['event'] == 1 else row['observation_time'], axis=1)
        

        features = df[config['feature_names']].to_numpy()
        
        labels = df[['time', 'event']].to_numpy()

        embeddings = df['embeddings']
        # apply json.loads to the embeddings
        embeddings = [json.loads(embedding) for embedding in embeddings]
        
        self.is_train = is_train
        self.data = []
        

        temp = []
        if config['use_embedding_input']:
            # Only use features and labels that have valid embeddings
            for i in range(len(features)):
                feature = torch.from_numpy(features[i]).float()
                time, event = labels[i][0], labels[i][1]
                time = int(time/config['discount_factor'])
                embedding = torch.tensor(embeddings[i], dtype=torch.float32)
                temp.append([time, event, feature, embedding])
        else:
            for i, (feature, label) in enumerate(zip(features, labels)):
                feature = torch.from_numpy(feature).float()
                time, event = label[0], label[1]
                time = int(time/config['discount_factor'])
                temp.append([time, event, feature])
                    
        sorted_temp = sorted(temp, key=itemgetter(0)) # sort by time/duration

        if self.is_train:
            new_temp = sorted_temp # only use the sorted data for training
        else:
            new_temp = temp

        if config['use_embedding_input']:
            for time, is_observed, feature, embedding in new_temp:
            # Convert duration to integer for list operations
                time_int = int(time)
                # Convert embedding to tensor if it's a list
                if isinstance(embedding, list):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                elif not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                
                if is_observed:
                    mask = config['max_seq_len'] * [1.]
                    label = time_int * [1.] + (config['max_seq_len'] - time_int) * [0.]
                    # feature = torch.stack(config['max_time'] * [feature])
                    self.data.append([feature.to(self.device), torch.tensor(time).float().to(self.device), torch.tensor(mask).float().to(self.device), torch.tensor(label).to(self.device), torch.tensor(is_observed).bool().to(self.device), embedding.to(self.device)])
                else:
                    # NOTE plus 1 to include day 0
                    mask = (time_int + 1) * [1.] + (config['max_seq_len'] - (time_int + 1)) * [0.]
                    label = config['max_seq_len'] * [1.]
                    # feature = torch.stack(config['max_time'] * [feature])
                    self.data.append([feature.to(self.device), torch.tensor(time).float().to(self.device), torch.tensor(mask).float().to(self.device), torch.tensor(label).to(self.device), torch.tensor(is_observed).bool().to(self.device), embedding.to(self.device)])
        else:
            for time, is_observed, feature in new_temp:
                # Convert duration to integer for list operations
                time_int = int(time)
                if is_observed:
                    mask = config['max_seq_len'] * [1.]
                    label = time_int * [1.] + (config['max_seq_len'] - time_int) * [0.]
                    # feature = torch.stack(config['max_time'] * [feature])
                    # Create a dummy embedding tensor when not using embeddings
                    dummy_embedding = torch.zeros(1, device=self.device)  # Dummy tensor instead of None
                    self.data.append([feature.to(self.device), torch.tensor(time).float().to(self.device), torch.tensor(mask).float().to(self.device), torch.tensor(label).to(self.device), torch.tensor(is_observed).bool().to(self.device), dummy_embedding])
                else:
                    # NOTE plus 1 to include day 0
                    mask = (time_int + 1) * [1.] + (config['max_seq_len'] - (time_int + 1)) * [0.]
                    label = config['max_seq_len'] * [1.]
                    # feature = torch.stack(config['max_time'] * [feature])
                    # Create a dummy embedding tensor when not using embeddings
                    dummy_embedding = torch.zeros(1, device=self.device)  # Dummy tensor instead of None
                    self.data.append([feature.to(self.device), torch.tensor(time).float().to(self.device), torch.tensor(mask).float().to(self.device), torch.tensor(label).to(self.device), torch.tensor(is_observed).bool().to(self.device), dummy_embedding])

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a+1, len(self.data))
            # print(len(self.data[index_a]))
            return [ [self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a])) ]
        else:
            return self.data[index_a]

    def __len__(self):
        return len(self.data)

class SurvivalDataSet_2021Transformer_pd(Dataset): # input is pandas dataframe

    def __init__(self, df, config, is_train=True):
        self.device = get_best_device()
        
        features = df[config['feature_names']].to_numpy()
        labels = df[['time', 'event']].to_numpy()
        use_embedding = config.get('use_embedding_input', False)
        if use_embedding:
            embeddings = df['embeddings']
            embeddings = [json.loads(embedding) for embedding in embeddings]
        else:
            embeddings = None
        self.is_train = is_train
        self.data = []
        

        temp = []
        if use_embedding:
            # Only use features and labels that have valid embeddings
            for i in range(len(features)):
                feature = torch.from_numpy(features[i]).float()
                time, event = labels[i][0], labels[i][1]
                time = int(time/config['discount_factor'])
                embedding = torch.tensor(embeddings[i], dtype=torch.float32)
                temp.append([time, event, feature, embedding])
        else:
            for i, (feature, label) in enumerate(zip(features, labels)):
                feature = torch.from_numpy(feature).float()
                time, event = label[0], label[1]
                time = int(time/config['discount_factor'])
                temp.append([time, event, feature])
                    
        sorted_temp = sorted(temp, key=itemgetter(0)) # sort by time/duration

        if self.is_train:
            new_temp = sorted_temp # only use the sorted data for training
        else:
            new_temp = temp

        if config['use_embedding_input']:
            for time, is_observed, feature, embedding in new_temp:
            # Convert duration to integer for list operations
                time_int = int(time)
                # Convert embedding to tensor if it's a list
                if isinstance(embedding, list):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                elif not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                
                if is_observed:
                    mask = config['max_seq_len'] * [1.]
                    label = time_int * [1.] + (config['max_seq_len'] - time_int) * [0.]
                    # feature = torch.stack(config['max_time'] * [feature])
                    self.data.append([feature.to(self.device), torch.tensor(time).float().to(self.device), torch.tensor(mask).float().to(self.device), torch.tensor(label).to(self.device), torch.tensor(is_observed).bool().to(self.device), embedding.to(self.device)])
                else:
                    # NOTE plus 1 to include day 0
                    mask = (time_int + 1) * [1.] + (config['max_seq_len'] - (time_int + 1)) * [0.]
                    label = config['max_seq_len'] * [1.]
                    # feature = torch.stack(config['max_time'] * [feature])
                    self.data.append([feature.to(self.device), torch.tensor(time).float().to(self.device), torch.tensor(mask).float().to(self.device), torch.tensor(label).to(self.device), torch.tensor(is_observed).bool().to(self.device), embedding.to(self.device)])
        else:
            for time, is_observed, feature in new_temp:
                # Convert duration to integer for list operations
                time_int = int(time)
                if is_observed:
                    mask = config['max_seq_len'] * [1.]
                    label = time_int * [1.] + (config['max_seq_len'] - time_int) * [0.]
                    # feature = torch.stack(config['max_time'] * [feature])
                    # Create a dummy embedding tensor when not using embeddings
                    dummy_embedding = torch.zeros(1, device=self.device)  # Dummy tensor instead of None
                    self.data.append([feature.to(self.device), torch.tensor(time).float().to(self.device), torch.tensor(mask).float().to(self.device), torch.tensor(label).to(self.device), torch.tensor(is_observed).bool().to(self.device), dummy_embedding])
                else:
                    # NOTE plus 1 to include day 0
                    mask = (time_int + 1) * [1.] + (config['max_seq_len'] - (time_int + 1)) * [0.]
                    label = config['max_seq_len'] * [1.]
                    # feature = torch.stack(config['max_time'] * [feature])
                    # Create a dummy embedding tensor when not using embeddings
                    dummy_embedding = torch.zeros(1, device=self.device)  # Dummy tensor instead of None
                    self.data.append([feature.to(self.device), torch.tensor(time).float().to(self.device), torch.tensor(mask).float().to(self.device), torch.tensor(label).to(self.device), torch.tensor(is_observed).bool().to(self.device), dummy_embedding])

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a+1, len(self.data))
            # print(len(self.data[index_a]))
            return [ [self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a])) ]
        else:
            return self.data[index_a]

    def __len__(self):
        return len(self.data)
