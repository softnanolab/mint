import pandas as pd
from multiprocessing import Pool
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle, yaml
from tqdm import tqdm
import os
from easydict import EasyDict

import os
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
import torch

import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, median_absolute_error, roc_auc_score
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from torch.utils.data import Subset

DATA_ROOT = './TEIM/data'

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def load_data(config):
    dataset_name = config.dataset
    split_type = getattr(config, 'split', '')
    if dataset_name == 'seqlevel_data':
        dataset = SeqLevelDataset(config)
        if split_type == 'cv-new_epitope':
            epi2cluster = get_cluster(dataset_name, split_type)
            epi_id = dataset.get_all_epiid()
            group_id = [epi2cluster[epi_this] for epi_this in epi_id]

            kfold = GroupKFold(5)
            splits = kfold.split(epi_id, groups=group_id)
            datasets_cv = {'train':[],'val':[]}
            for train_idx, val_idx in splits:
                datasets_cv['train'].append(Subset(dataset, train_idx))
                datasets_cv['val'].append(Subset(dataset, val_idx))
            return datasets_cv

    elif dataset_name == 'reslevel_data':
        dataset = ResLevelDataset(config)
        if split_type in ['cv-both_new', 'cv-new_cdr3', 'cv-new_epi']:
            cdr32cluster, epi2cluster = get_cluster(dataset_name, split_type)

            kfold = GroupKFold(3) if split_type == 'cv-both_new' else GroupKFold(5)
            if len(cdr32cluster) != 0:  # new_cdr3, both_new
                all_cdr3= dataset.get_all_values('cdr3_seqs')
                cdr3_group_id = [cdr32cluster[cdr3_this] for cdr3_this in all_cdr3]
                split_cdr3 = list(kfold.split(all_cdr3, groups=cdr3_group_id))
                split = split_cdr3
            if len(epi2cluster) != 0:  # new_epi, both_new
                all_epi = dataset.get_all_values('epi_seqs')
                epi_group_id = [epi2cluster[epi_this] for epi_this in all_epi]
                split_epi = list(kfold.split(all_epi, groups=epi_group_id))
                split = split_epi
            if split_type == 'cv-both_new':
                split = [[np.intersect1d(fold_tcr[0], fold_epi[0]), np.intersect1d(fold_tcr[1], fold_epi[1])]
                    for fold_tcr, fold_epi in zip(split_cdr3, split_epi)]
            
            datasets_cv = {'train':[],'val':[]}
            for train_idx, val_idx in split:
                datasets_cv['val'].append(Subset(dataset, val_idx))
                datasets_cv['train'].append(Subset(dataset, train_idx))
            return datasets_cv

    if (split_type is None) or (split_type == ''):
        print('No split specified, using train as default')
        return dataset
    elif split_type == 'train-val':
        train_ratio = getattr(config, 'train_ratio', 0.8)
        index = np.random.permutation(len(dataset))
        train_set = Subset(dataset, index[:int(len(dataset) * train_ratio)])
        val_set = Subset(dataset, index[int(len(dataset) * train_ratio):])
        return {'train': [train_set], 'val': [val_set]}
    else:
        raise ValueError('Unknown split: {}'.format(split_type))

def get_cluster(dataset_name, split):
    if dataset_name == 'seqlevel_data' and split == 'cv-new_epitope':
        cluster_path = os.path.join(DATA_ROOT, 'cluster/seqlevel_epi_cluster_0.5.pkl')
        with open(cluster_path, 'rb') as f:
            epi2cluster = pickle.load(f, encoding='iso-8859-1')
        epi2cluster = {int(k.split('_')[-1]):v for k, v in epi2cluster.items()}
        return epi2cluster
    if dataset_name == 'reslevel_data':
        cluster_path_dict = {
            'cdr3': os.path.join(DATA_ROOT, 'cluster/reslevel_cdr3_cluster_0.2.pkl'),
            'epi': os.path.join(DATA_ROOT, 'cluster/reslevel_epi_cluster_0.2.pkl'),
        }
        cdr32cluster, epi2cluster = {}, {}
        if ('new_cdr3' in split) or ('both_new' in split):
            with open(cluster_path_dict['cdr3'], 'rb') as f:
                cdr32cluster = pickle.load(f, encoding='iso-8859-1')
        if ('new_epi' in split) or ('both_new' in split):
            with open(cluster_path_dict['epi'], 'rb') as f:
                epi2cluster = pickle.load(f, encoding='iso-8859-1')
        return cdr32cluster, epi2cluster

class SeqLevelDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        path = config.path
        file_list = config.file_list
        data = self.load_data(path, file_list)
        negative = getattr(config, 'negative', 'original')
        if negative == 'original':
            print('original negative samples')
        elif negative == 'shuffle':
            print('shuffle negative samples')
            data = self.make_shuffled_nega(data)
        self.data = self.encoding(data)

        # add baseline encoding
        baseline = getattr(config, 'baseline', None)
        if baseline is not None:
            print('Add baseline encoding', baseline)
            self.data = self.add_encoding(self.data, baseline)

    def add_encoding(self, data, baseline):
        return data


    def load_data(self, path, file_list):
        for file in file_list:
            try:
                df = pd.read_csv(os.path.join(path, file+'.tsv'), sep='\t')
                data_this = df[['cdr3', 'epitope', 'label']].values
            except FileNotFoundError:
                df = pd.read_csv(os.path.join(path, file+'.csv'))
                data_this = df[['cdr3', 'epi', 'y_true']].values

            if 'data' not in locals():
                data = [[] for _ in range(len(data_this[0]))]
            for i in range(len(data_this[0])):
                data[i].extend(data_this[:, i])
        
        # load epitope id
        try:
            df_epi = pd.read_csv(os.path.join(path, 'positive_epi_dist.tsv'), sep='\t', index_col=0)
            df_epi = df_epi['Epitope_idx']
            epi_ids = df_epi.loc[data[1]].values
        except FileNotFoundError:
            print('epi ID if not found!')
            epi_ids = - np.ones(len(data[1]))
        data = data + [epi_ids]

        # to dict
        data = {
            'cdr3': np.array(data[0]),
            'epi': np.array(data[1]),
            'labels': np.array(data[2]),
            'epi_id': np.array(data[3])
        }
        return data

    def encoding(self, data):
        cdr3, epi = data['cdr3'], data['epi']
        with Pool(processes=64) as p:
            enc_cdr3 = p.map(encoding_cdr3_single, cdr3)
            enc_epi = p.map(encoding_epi_single, epi)
        # enc_cdr3 = encoding_cdr3(cdr3)
        # enc_epi = encoding_epi(epi)
        data['cdr3'] = np.array(enc_cdr3)
        data['epi'] = np.array(enc_epi)
        data['cdr3_seqs'] = cdr3
        data['epi_seqs'] = epi
        
        return data

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        return {key:value[idx] for key, value in self.data.items()}
        # return {
        #     'cdr3': self.data['cdr3'][idx],
        #     'epi': self.data['epi'][idx],
        #     'labels': self.data['labels'][idx],
        #     'epi_id': self.data['epi_id'][idx],
        #     'cdr3_seqs': self.data['cdr3_seqs'][idx],
        #     'epi_seqs': self.data['epi_seqs'][idx],
        # }

    def get_all_epiid(self):
        return self.data['epi_id']

    def make_shuffled_nega(self, data):
        cdr3 = data['cdr3']
        epi = data['epi']
        labels = data['labels']
        epi_id = data['epi_id']
        # get out positive samples  
        ind_pos = (labels == 1)
        cdr3_pos = cdr3[ind_pos]
        epi_pos = epi[ind_pos]
        labels_pos = labels[ind_pos]
        epi_id_pos = epi_id[ind_pos]
        # prepare info
        ratio = 5  # (len(ind_pos) - ind_pos.sum()) /ind_pos.sum()
        df_pos = pd.DataFrame({'cdr3': cdr3_pos, 'epi': epi_pos, 'labels':labels_pos, 'epi_id':epi_id_pos})
        epi2id = df_pos[['epi', 'epi_id']].set_index('epi').to_dict()['epi_id']

        # make negative by shuffling
        df_all = pd.pivot(df_pos, index='epi', columns='cdr3', values='labels')
        df_all = df_all.fillna(0)
        df_all = df_all.unstack().reset_index().rename(columns={0: 'labels'})
        df_all_shuffled = df_all[df_all['labels']==0]
        epi_counts = df_pos['epi'].value_counts()
        # sample negatives
        df_new_neg_list = []
        for epi, counts_pos in tqdm(epi_counts.items(), total=len(epi_counts), desc='sampling negatives'):
            try:
                df_new_neg = df_all_shuffled[df_all_shuffled['epi']==epi].sample(int(counts_pos * ratio))
            except ValueError:
                df_new_neg = df_all_shuffled[df_all_shuffled['epi']==epi].sample(int(counts_pos * ratio), replace=True)
            df_new_neg_list.append(df_new_neg)
        df_new_neg = pd.concat(df_new_neg_list)
        cdr3_neg = df_new_neg['cdr3']
        epi_neg = df_new_neg['epi']
        labels_neg = df_new_neg['labels']
        epi_id_neg = np.array([epi2id[epi] for epi in epi_neg])
        # concat
        cdr3_all = np.concatenate([cdr3_pos, cdr3_neg], axis=0)
        epi_all = np.concatenate([epi_pos, epi_neg], axis=0)
        labels_all = np.concatenate([labels_pos, labels_neg], axis=0)
        epi_id_all = np.concatenate([epi_id_pos, epi_id_neg], axis=0)
        return {
            'cdr3': cdr3_all,
            'epi': epi_all,
            'labels': labels_all,
            'epi_id': epi_id_all,
        }


class ResLevelDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        path = config.path
        data = self.load_data(path)
        self.data = self.encoding(data)


    def load_data(self, path):
        df = pd.read_csv(os.path.join(path['summary']))

        cdr3_seqs = df['cdr3'].values
        epi_seqs = df['epitope'].values
        pdb_chains = df['pdb_chains'].values
        pdb_mat = []
        for pdb in pdb_chains:
            df_mat = pd.read_csv(os.path.join(path['mat'], pdb + '.csv'), index_col=0)
            pdb_mat.append(df_mat.values)
        return {
            'cdr3': cdr3_seqs,
            'epi': epi_seqs,
            'dist_mat': pdb_mat,
            'pdb_chains': pdb_chains,
        }

    def encoding(self, data):
        cdr3, epi, mat = data['cdr3'], data['epi'], data['dist_mat']
        # with Pool(processes=64) as p:
        #     enc_cdr3 = p.map(encoding_cdr3_single, cdr3)
        #     enc_epi = p.map(encoding_epi_single, epi)
        enc_cdr3 = encoding_cdr3(cdr3)
        enc_epi = encoding_epi(epi)
        enc_dist_mat, masking = encoding_dist_mat(mat)
        enc_contact_mat = np.int64(enc_dist_mat < 5.)

        data['cdr3'] = np.array(enc_cdr3)
        data['epi'] = np.array(enc_epi)
        data['dist_mat'] = np.array(enc_dist_mat)
        data['mask_mat'] = np.array(masking)
        data['contact_mat'] = np.array(enc_contact_mat)
        data['cdr3_seqs'] = cdr3
        data['epi_seqs'] = epi

        return data

    def __len__(self):
        return len(self.data['dist_mat'])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}

        
    def get_all_values(self, key):
        return self.data[key]



def encoding_dist_mat(mat_list, max_cdr3=20, max_epi=12):
    encoding = np.zeros([len(mat_list), max_cdr3, max_epi], dtype='float32')
    masking = np.zeros([len(mat_list), max_cdr3, max_epi], dtype='bool')
    for i, mat in tqdm(enumerate(mat_list), desc='Encoding dist mat', total=len(mat_list)):
        len_cdr3, len_epi = mat.shape
        i_start_cdr3 = max_cdr3 // 2 - len_cdr3 // 2
        if len_epi == 8:
            i_start_epi = 2
        elif (len_epi == 9) or (len_epi == 10):
            i_start_epi = 1
        else:
            i_start_epi = 0
        encoding[i, i_start_cdr3:i_start_cdr3+len_cdr3, i_start_epi:i_start_epi+len_epi] = mat
        masking[i, i_start_cdr3:i_start_cdr3+len_cdr3, i_start_epi:i_start_epi+len_epi] = True
    return encoding, masking

def calc_auc_aupr(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    return auc, aupr

def get_scores_dist(y_true, y_pred, y_mask):
    coef, mae, mape = [], [], []
    for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):
        y_true_ = y_true_[y_mask_.astype('bool')]
        y_pred_ = y_pred_[y_mask_.astype('bool')]
        try:
            coef_, _ = pearsonr(y_true_, y_pred_)
        except Exception:
            coef_ = np.nan
        coef.append(coef_)

        mae_ = median_absolute_error(y_true_, y_pred_)
        mae.append(mae_)

        mape_ = np.median(np.abs((np.array(y_true_) - np.array(y_pred_)) / np.array(y_true_)))
        mape.append(mape_)
    avg_coef = np.nanmean(coef)
    avg_mae = np.nanmean(mae)
    avg_mape = np.nanmean(mape)
    return [avg_coef, avg_mae, avg_mape], [coef, mae, mape]


def get_scores_contact(y_true, y_pred, y_mask):
    roc_auc_scores = []
    auprc_scores = []
    for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):
        y_true_ = y_true_[y_mask_.astype('bool')]
        y_pred_ = y_pred_[y_mask_.astype('bool')]
        
        try:
            roc_auc = roc_auc_score(y_true_, y_pred_)
        except:
            roc_auc = np.nan
            
        try:
            auprc = average_precision_score(y_true_, y_pred_)
        except:
            auprc = np.nan

        roc_auc_scores.append(roc_auc)
        auprc_scores.append(auprc)
    avg_roc_auc = np.nanmean(roc_auc_scores)
    avg_auprc = np.nanmean(auprc_scores)
    
    return [avg_roc_auc, avg_auprc], [roc_auc_scores, auprc_scores]

def decoding_one_mat(mat, len_cdr3, len_epi):
    decoding = np.zeros([len_cdr3, len_epi] + list(mat.shape[2:]), dtype=mat.dtype)
    i_start_cdr3 = 10 - len_cdr3 // 2
    if len_epi == 8:
        i_start_epi = 2
    elif (len_epi == 9) or (len_epi == 10):
        i_start_epi = 1
    else:
        i_start_epi = 0
    decoding = mat[i_start_cdr3:i_start_cdr3+len_cdr3, i_start_epi:i_start_epi+len_epi] 
    return decoding

def GetBlosumMat(residues_list):
    n_residues = len(residues_list)  # the number of amino acids _ 'X'
    blosum62_mat = np.zeros([n_residues, n_residues])  # plus 1 for gap
    bl_dict = substitution_matrices.load('BLOSUM62')
    for pair, score in bl_dict.items():
        if (pair[0] not in residues_list) or (pair[1] not in residues_list):  # special residues not considered here
            continue
        idx_pair0 = residues_list.index(pair[0])  # index of residues
        idx_pair1 = residues_list.index(pair[1])
        blosum62_mat[idx_pair0, idx_pair1] = score
        blosum62_mat[idx_pair1, idx_pair0] = score
    return blosum62_mat


class Tokenizer:
    def __init__(self,):
        self.res_all = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'N',
                     'E', 'K', 'Q', 'M', 'S', 'T', 'C', 'P', 'H', 'R'] #+ ['X'] #BJZOU
        self.tokens = ['-'] + self.res_all # '-' for padding encoding

    def tokenize(self, index): # int 2 str
        return self.tokens[index]

    def id(self, token): # str 2 int
        try:
            return self.tokens.index(token.upper())
        except ValueError:
            print('Error letter in the sequences:', token)
            if str.isalpha(token):
                return self.tokens.index('X')

    def tokenize_list(self, seq):
        return [self.tokenize(i) for i in seq]

    def id_list(self, seq):
        return [self.id(s) for s in seq]

    def embedding_mat(self):
        blosum62 = GetBlosumMat(self.res_all)
        mat = np.eye(len(self.tokens))
        mat[1:len(self.res_all) + 1, 1:len(self.res_all) + 1] = blosum62
        return mat

tokenizer = Tokenizer()


def encoding_epi(seqs, max_len=12):
    encoding = np.zeros([len(seqs), max_len], dtype='long')
    for i, seq in tqdm(enumerate(seqs), desc='Encoding epi seqs', total=len(seqs)):
        len_seq = len(seq)
        if len_seq == 8:
            encoding[i, 2:len_seq+2] = tokenizer.id_list(seq)
        elif (len_seq == 9) or (len_seq == 10):
            encoding[i, 1:len_seq+1] = tokenizer.id_list(seq)
        else:
            encoding[i, :len_seq] = tokenizer.id_list(seq)
    return encoding

def encoding_cdr3(seqs, max_len=20):
    encoding = np.zeros([len(seqs), max_len], dtype='long')
    for i, seq in tqdm(enumerate(seqs), desc='Encoding cdr3s', total=len(seqs)):
        len_seq = len(seq)
        i_start =  max_len // 2 - len_seq // 2
        encoding[i, i_start:i_start+len_seq] = tokenizer.id_list(seq)
    return encoding

def encoding_cdr3_single(seq, max_len=20):
    encoding = np.zeros(max_len, dtype='long')
    len_seq = len(seq)
    i_start =  max_len // 2 - len_seq // 2
    encoding[i_start:i_start+len_seq] = tokenizer.id_list(seq)
    return encoding

def encoding_epi_single(seq, max_len=12):
    encoding = np.zeros(max_len, dtype='long')
    len_seq = len(seq)
    if len_seq == 8:
        encoding[2:len_seq+2] = tokenizer.id_list(seq)
    elif (len_seq == 9) or (len_seq == 10):
        encoding[1:len_seq+1] = tokenizer.id_list(seq)
    else:
        encoding[:len_seq] = tokenizer.id_list(seq)
    return encoding


def load_ae_model(tokenizer, path='./ckpt/epi_ae.ckpt'):
    # tokenizer = Tokenizer()
    ## load model
    model_args = dict(
        tokenizer = tokenizer,
        dim_hid = 32,
        len_seq = 12,
    )
    model = AutoEncoder(**model_args)
    model.eval()

    ## load weights
    state_dict = torch.load(path)
    state_dict = {k[6:]:v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


class PretrainedEncoder:
    def __init__(self, tokenizer):
        self.ae_model = load_ae_model(tokenizer)
        self.tokenizer = tokenizer

    def encode_pretrained_epi(self, epi_seqs):
        enc = self.infer(epi_seqs)
        enc_vec = enc[2]
        enc_seq = enc[-1]
        return enc_seq, enc_vec
    
    def infer(self, seqs):
        # # seqs encoding
        n_seqs = len(seqs)
        len_seqs = [len(seq) for seq in seqs]
        assert (np.max(len_seqs) <= 12) and (np.min(len_seqs)>=8), ValueError('Lengths of epitopes must be within [8, 12]')
        encoding = np.zeros([n_seqs, 12], dtype='int32')
        for i, seq in enumerate(seqs):
            len_seq = len_seqs[i]
            if len_seq == 8:
                encoding[i, 2:len_seq+2] = self.tokenizer.id_list(seq)
            elif (len_seq == 9) or (len_seq == 10):
                encoding[i, 1:len_seq+1] = self.tokenizer.id_list(seq)
            else:
                encoding[i, :len_seq] = self.tokenizer.id_list(seq)
        # # pretrained ae features
        inputs = torch.from_numpy(encoding)
        out, seq_enc, vec, indices = self.ae_model(inputs)
        out = np.argmax(out.detach().cpu().numpy(), -1)
        return [
            out,
            seq_enc.detach().cpu().numpy(),
            vec.detach().cpu().numpy(),
            indices,
            encoding
        ]


def encode_cdr3(cdr3, tokenizer):
    len_cdr3 = [len(s) for s in cdr3]
    max_len_cdr3 = np.max(len_cdr3)
    assert max_len_cdr3 <= 20, 'The cdr3 length must <= 20'
    max_len_cdr3 = 20
    
    seqs_al = get_numbering(cdr3)
    num_samples = len(seqs_al)

    # encoding
    encoding_cdr3 = np.zeros([num_samples, max_len_cdr3], dtype='int32')
    for i, seq in enumerate(seqs_al):
        encoding_cdr3[i, ] = tokenizer.id_list(seq)
    return encoding_cdr3


def encode_epi(epi, tokenizer):
    # tokenizer = Tokenizer()
    encoding_epi = np.zeros([12], dtype='int32')
    len_epi = len(epi)
    if len_epi == 8:
        encoding_epi[2:len_epi+2] = tokenizer.id_list(epi)
    elif (len_epi == 9) or (len_epi == 10):
        encoding_epi[1:len_epi+1] = tokenizer.id_list(epi)
    else:
        encoding_epi[:len_epi] = tokenizer.id_list(epi)
    return encoding_epi

class AutoEncoder(nn.Module):
    def __init__(self, 
        dim_hid,
        len_seq,
    ):
        super().__init__()
        embedding = tokenizer.embedding_mat()
        vocab_size, dim_emb = embedding.shape
        self.embedding_module = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), padding_idx=0, )
        self.encoder = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Conv1d(dim_hid, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )

        self.seq2vec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len_seq * dim_hid, dim_hid),
            nn.ReLU()
        )
        self.vec2seq = nn.Sequential(
            nn.Linear(dim_hid, len_seq * dim_hid),
            nn.ReLU(),
            View(dim_hid, len_seq)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(dim_hid, vocab_size)

    def forward(self, inputs, latent_only=False):
        # inputs = inputs.long()
        seq_emb = self.embedding_module(inputs)
        seq_enc = self.encoder(seq_emb.transpose(1, 2))
        vec = self.seq2vec(seq_enc)
        seq_repr = self.vec2seq(vec)
        seq_dec = self.decoder(seq_repr)
        out = self.out_layer(seq_dec.transpose(1, 2))
        if latent_only:
            return vec
        else:
            return out, seq_enc, vec

def load_model_from_ckpt(path, model: torch.nn.Module):
    pl_model = torch.load(path)
    if 'state_dict' in pl_model:
        pl_model = pl_model['state_dict'] # from lightning module
    state_dict = {k[k.find('.')+1:]: v for k, v in pl_model.items()}
    model.load_state_dict(state_dict)
    return model
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        shape = [input.shape[0]] + list(self.shape)
        return input.view(*shape)
