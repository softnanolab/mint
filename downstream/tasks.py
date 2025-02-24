import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def clean_seq(seq):
    seq = seq.replace("*", "")
    return seq.replace('f', '')

class CSVDataset(Dataset):
    def __init__(
        self, 
        df_path,
        col1, 
        col2,
        target_col,
        test_run
    ):
        super().__init__()
        self.df = pd.read_csv(df_path)

        if test_run:
            self.df = self.df.sample(n=20)

        self.seqs1 = self.df[col1].tolist()
        self.seqs2 = self.df[col2].tolist()
        if type(target_col) == list:
            self.targets = self.df[target_col].to_numpy()
        else:
            self.targets = self.df[target_col].tolist()
        
    def __len__(self):
        return len(self.seqs1)

    def __getitem__(self, index):
        return clean_seq(self.seqs1[index]), clean_seq(self.seqs2[index]), self.targets[index]

class MutationalCSVDataset(Dataset):
    def __init__(
        self, 
        df_path,
        col1, 
        col2,
        col3, 
        col4,
        target_col,
        test_run
    ):
        super().__init__()
        self.df = pd.read_csv(df_path)

        if test_run:
            self.df = self.df.sample(n=20)

        self.seqs1 = self.df[col1].tolist()
        self.seqs2 = self.df[col2].tolist()
        self.seqs3 = self.df[col3].tolist()
        self.seqs4 = self.df[col4].tolist()
        if type(target_col) == list:
            self.targets = self.df[target_col].to_numpy()
        else:
            self.targets = self.df[target_col].tolist()
        
    def __len__(self):
        return len(self.seqs1)

    def __getitem__(self, index):
        return self.seqs1[index], self.seqs2[index], self.seqs3[index], self.seqs4[index], self.targets[index]

class MultiCSVDataset(Dataset):
    def __init__(
        self, 
        df_path,
        col1, 
        col2,
        target_col,
        test_run
    ):
        super().__init__()
        self.df = pd.read_csv(df_path)

        if test_run:
            self.df = self.df.sample(n=20)

        self.seqs = self.df[col1].tolist()
        self.chain_ids = self.df[col2].tolist()
        if type(target_col) == list:
            self.targets = self.df[target_col].to_numpy()
        else:
            self.targets = self.df[target_col].tolist()
        
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        chain_ids = torch.tensor([int(i) for i in self.chain_ids[index]])
        return self.seqs[index], chain_ids, self.targets[index]


def join_sequences(seqs1, seqs2):
    return [s1 + s2 for s1,s2 in zip(seqs1,seqs2)]

def get_task_datasets(task, test_run=False, return_metadata=False):
    if task == 'HumanPPI':
        train_fl = './human-ppi/processed_data_train.csv'
        val_fl = './human-ppi/processed_data_validation.csv'
        test_fl = './human-ppi/processed_data_test.csv'
        task_type = 'bc'
        col1 = 'sequence_1'
        col2 = 'sequence_2'
        target_col = 'target'
        num_epochs = 100
        method = 'mlp'
        output_size = 1
        monitor_metric = 'Accuracy'
    elif task == 'YeastPPI':
        train_fl = './yeast-ppi/processed_data_train.csv'
        val_fl = './yeast-ppi/processed_data_validation.csv'
        test_fl = './yeast-ppi/processed_data_test.csv'
        task_type = 'bc'   
        col1 = 'sequence_1'
        col2 = 'sequence_2'
        target_col = 'target'
        num_epochs = 100
        method = 'mlp'
        output_size = 1
        monitor_metric = 'Accuracy'
    elif task == 'SKEMPI':
        train_fl = './SKEMPI_v2/processed_data.csv'
        val_fl = None
        test_fl = None
        task_type = 'reg'   
        col1 = 'seq1'
        col2 = 'seq2'
        mut_col1 = 'seq1_mut'
        mut_col2 = 'seq2_mut'
        target_col = 'target'
        num_epochs = 20
        method = 'pcv'
        output_size = None
        monitor_metric = None
    elif task == 'Bernett':
        train_fl = './ppi/Intra1_seqs.csv'
        val_fl = './ppi/Intra0_seqs.csv'
        test_fl = './ppi/Intra2_seqs.csv'
        task_type = 'bc'   
        col1 = 'seq1'
        col2 = 'seq2'
        target_col = 'labels'
        num_epochs = 30
        method = 'mlp'
        output_size = 1
        monitor_metric = 'AUPRC'
    elif task == 'Pdb-bind':
        train_fl = './pdb-bind/processed_data.csv'
        val_fl = None
        test_fl = None
        task_type = 'reg'   
        col1 = 'seq'
        col2 = 'chain_ids'
        target_col = 'target' 
        num_epochs = 20
        method = 'cv'
        output_size = None
        monitor_metric = None
    elif task == 'MutationalPPI':
        train_fl = './mutants/processed_data.csv'
        val_fl = None
        test_fl = None
        task_type = 'bc'   
        col1 = 'seq1'
        col2 = 'seq2'
        target_col = 'target' 
        num_epochs = None
        method = 'cv'
        output_size = None
        monitor_metric = None
    elif task == 'MutationalPPI_cs':
        train_fl = './mutants/processed_data_cs.csv'
        val_fl = './mutants/processed_data_val_cs.csv'
        test_fl = './mutants/processed_data_test_cs.csv'
        task_type = 'bc'   
        col1 = 'seq1'
        col2 = 'seq2'
        mut_col1 = 'seq1_mut'
        mut_col2 = 'seq2_mut'
        target_col = 'target' 
        num_epochs = None
        method = 'cv'
        output_size = None
        monitor_metric = None
    # elif task == 'crispr':
    #     train_fl = './crispr/processed_data_train.csv'
    #     val_fl = None
    #     test_fl = './crispr/processed_data_test.csv'
    #     task_type = 'reg'   
    #     col1 = 'seq'
    #     col2 = 'chain_ids'
    #     target_col = 'target' 
    #     num_epochs = 20
    #     method = 'cv'
    #     output_size = None
    #     monitor_metric = None
    else:
        return 0

    if task == 'SKEMPI' or task == 'MutationalPPI_cs':
        train_dataset = MutationalCSVDataset(train_fl, col1, col2, mut_col1, mut_col2, target_col, test_run)
        if val_fl is not None:
            val_dataset = MutationalCSVDataset(val_fl, col1, col2, mut_col1, mut_col2, target_col, test_run)
        else:
            val_dataset = None
        if test_fl is not None:
            test_dataset = MutationalCSVDataset(test_fl, col1, col2, mut_col1, mut_col2, target_col, test_run)
        else:
            test_dataset = None  
    elif task == 'Pdb-bind' or task == 'crispr':
        train_dataset = MultiCSVDataset(train_fl, col1, col2, target_col, test_run)
        val_dataset = MultiCSVDataset(val_fl, col1, col2, target_col, test_run) if val_fl != None else None
        test_dataset = MultiCSVDataset(test_fl, col1, col2, target_col, test_run) if test_fl != None else None
    else:
        train_dataset = CSVDataset(train_fl, col1, col2, target_col, test_run)
        if val_fl is not None:
            val_dataset = CSVDataset(val_fl, col1, col2, target_col, test_run)
        else:
            val_dataset = None
        if test_fl is not None:
            test_dataset = CSVDataset(test_fl, col1, col2, target_col, test_run)
        else:
            test_dataset = None   

    if not return_metadata:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, val_dataset, test_dataset, {'task_type': task_type, 
                                                         'num_epochs': num_epochs,
                                                         'lr': 1e-4, 
                                                         'method': method, 
                                                         'output_size': output_size, 
                                                         'train_df': train_dataset.df, 
                                                         'monitor_metric': monitor_metric}
    