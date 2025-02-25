## Comprehensive benchmarking of ESM-MULTIMER on PPI prediction (Gold-standard dataset from Bernett et al.)

* Follow `prepare_data.ipynb` to download and process the dataset splits. 
* Use the `../embeddings_esm_multimer.py` and `../finetune_general.py` scripts to evaluate the model on this dataset. 

Note: You might have to change the location of the dataset splits in `../tasks.py` depending on where you stored them. 