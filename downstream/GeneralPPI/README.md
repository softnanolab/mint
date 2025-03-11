## Comprehensive benchmarking of MINT on PPI prediction

This folder contains information on running baseline PLMs and MINT on six different PPI prediction tasks. 

Note: We choose to cache the embeddings from each model on every input data point across all tasks (using `embeddings_mint.py` and `embeddings_baselines.py`) instead of re-running the models for every experimental repeat. 

### Baseline models

1. It is recommended to create a new virtual environment for running the baseline models due to version conflicts in the `huggingface` package, since MINT uses an older version. 
2. Next from this directory, run the following to download the baseline models:

```
from baselines import ProtT5, ProGen, ESM

model = ProtT5(model_name='Rostlab/prot_t5_xl_bfd', layers='last', devices=[0], batch_size=2)
model = ProtT5(model_name='Rostlab/prot_t5_xl_uniref50', layers='last', devices=[0], batch_size=2)
model = ProGen(model_name='hugohrban/progen2-xlarge', layers='last', devices=[0], batch_size=2)
model = ProGen(model_name='hugohrban/progen2-large', layers='last', devices=[0], batch_size=2)
model = ESM(model_name='facebook/esm2_t36_3B_UR50D', layers='last', devices=[0], batch_size=2)
model = ESM(model_name='facebook/esm2_t33_650M_UR50D', layers='last', devices=[0], batch_size=2)
model = ESM(model_name='facebook/esm2_t30_150M_UR50D', layers='last', devices=[0], batch_size=2)
model = ESM(model_name='facebook/esm1b_t33_650M_UR50S', layers='last', devices=[0], batch_size=2)
```

3. Now, you are ready to generate embeddings from protein sequence inputs. Run `embeddings_baselines.py` along with the model name  (Example: `facebook/esm2_t33_650M_UR50D`, see above point) and task. The available task names are HumanPPI, YeastPPI, SKEMPI, Bernett (Gold-standard PPI), Pdb-bind and MutationalPPI. This will create new folders corresponding containing the embeddings. The number of files created depends on whether the task has training, validation and test splits. 
4. Finally, execute `finetune_general.py` with the appropriate model name and task, along with hyperparameters that control the downstream prediction head. 

### MINT

1. Update the `CONFIG_DICT_PATH` variable in `embeddings_mint.py` to the location on your device. 
2. Run `embeddings_mint.py` with the `--checkpoint_path` flag that corresponds to the location of the trained MINT weights and the task. This will create new folders corresponding containing the embeddings. The number of files created depends on whether the task has training, validation and test splits. 
3. Run `finetune_general.py` with the `model` flag set to `mint`. 
