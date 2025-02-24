import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import Dataset, WeightedRandomSampler
from torch import nn
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit

class MLPDataset(Dataset):
    def __init__(
        self, 
        x, y
    ):
        super().__init__()
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMLP, self).__init__()

        self.project = torch.nn.Sequential(
                nn.Linear(input_size, input_size//2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(input_size//2, output_size)
            )
    def forward(self, x):
        return self.project(x)


def classification_metrics(targets, predictions, threshold=0.5):
    binary_predictions = (predictions >= threshold).astype(int)
    accuracy = accuracy_score(targets, binary_predictions)
    f1 = f1_score(targets, binary_predictions)
    auc_score = roc_auc_score(targets, predictions)
    precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions)
    auprc = auc(recall_vals, precision_vals)
    return {
        'Accuracy': accuracy,
        'AUPRC': auprc,
        'F1 Score': f1,
        'AUROC': auc_score,
    }


@torch.no_grad()
def evaluate(model, loader, name, device):
    preds = []
    targets = []
    for step, eval_batch in enumerate(loader):
        embs, target = eval_batch
        embs = embs.to(device)
        target = target.to(device)
        pred = model(embs).squeeze(-1)  
        pred = torch.sigmoid(pred)
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    metrics = classification_metrics(targets, preds)
    return metrics, preds

def train_mlp(train_loader, test_loader, epochs, lr, device, seed):
    torch.manual_seed(seed)
    model = SimpleMLP(1280*2, 1).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))

    best_test_f1 = 0
    best_preds = {}

    for epoch in range(epochs):
        loss_accum = 0
        for step, train_batch in enumerate(train_loader):

            model.train()
            optimizer.zero_grad()

            embs, target = train_batch
            embs = embs.to(device)
            target = target.to(device)

            pred = model(embs)   
            loss = loss_fn(pred.squeeze(), target.float())

            loss.backward()
            optimizer.step()
            
            loss_accum += loss.detach().cpu().item()

        test_metric, test_preds = evaluate(model, test_loader, 'test', device)
        #val_metric, val_preds = evaluate(model, val_loader, 'val', device)

        if test_metric['F1 Score'] > best_test_f1:
            best_test_f1 = test_metric['F1 Score']
            best_preds = test_preds
        scheduler.step(loss_accum/(step+1))
    return best_test_f1, best_preds
        

def convert_train_test_features(train, test):
    t = StandardScaler()
    train = t.fit_transform(train)
    test = t.transform(test)
    return train, test

def process_embs(m1, m2):
    return np.concatenate([m1-m2, m1*m2], -1)

def get_embeddings(model_name):
    train_embs = torch.load(f'../embeddings/MutationalPPI_cs/{model_name}/train.pt').squeeze().numpy()
    #train_embs = train_embs[:,0:1280] - train_embs[:,1280:]
    train_embs = process_embs(train_embs[:,0:1280], train_embs[:,1280:])

    test_embs = torch.load(f'../embeddings/MutationalPPI_cs/{model_name}/val.pt').squeeze().numpy()
    #test_embs = test_embs[:,0:1280] - test_embs[:,1280:]
    test_embs = process_embs(test_embs[:,0:1280], test_embs[:,1280:])
    
    return train_embs, test_embs

train, test = get_embeddings('esm-m-f-140')

train_targets = pd.read_csv('processed_data_cs.csv')['target'].values
test_targets = pd.read_csv('processed_data_val_cs.csv')['target'].values

num_zeros = (train_targets == 0).sum().item()
num_ones = (train_targets == 1).sum().item()
weights = torch.tensor([num_zeros, num_ones])
weights = 1/weights
samples_weight = torch.tensor([weights[t] for t in train_targets.astype(int)]).double()
num_to_draw = 2*num_ones
sampler = WeightedRandomSampler(samples_weight, num_to_draw, replacement=False)


train_dataset = MLPDataset(train, train_targets)
test_dataset = MLPDataset(test, test_targets)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, sampler=sampler
)

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=32, shuffle=True
# )
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

total_stop = 100

all_preds = []
random_seed = 0
stop = 0
while stop==0:
    best_test_f1, best_preds = train_mlp(train_loader, test_loader, 50, 1e-5, 'cuda:6', random_seed)
    all_preds.append(best_preds)
    random_seed = random_seed + 1
    if len(all_preds) == total_stop:
        stop = 1

all_preds = np.array(all_preds)
np.save(f'best_preds_{total_stop}_rep.npy', all_preds)
