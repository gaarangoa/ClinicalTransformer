import os
import numpy as np
from tqdm import tqdm
import torch
import anndata as ad
from scipy.sparse import csr_matrix
from anndata.experimental.multi_files import AnnCollection

def get_embeddings(model, data, sample_ids, preprocessor, device, mapper={}, path=None, **kwargs):
    """
    Generates and collects model embeddings from a dataset, returning them as a concatenated AnnData object.

    This function iterates over a dataset, processes each sample using the provided model to extract embeddings,
    and stores the results in individual AnnData objects with structured observation and variable metadata.
    Finally, it concatenates all AnnData objects into a single AnnData dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model used to generate embeddings. Should accept tokens and values as input and produce hidden states.
    data : torch.utils.data.Dataset
        The dataset to process, where each item returns a tuple of (tokens, values, label).
    sample_ids : list or np.ndarray
        A list or array of sample IDs corresponding to the dataset. These IDs are used to prefix observation names
        and to store patient IDs in the AnnData object.
    preprocessor : object
        An object for token decoding, typically providing access to `preprocessor.feature_decoder`.
    device : torch.device or str
        The device on which computation is performed (e.g., "cpu" or "cuda").
    mapper : dict, optional
        A mapping from feature names to alternative names; applied for creating token_name2 in observations. 
        Defaults to an empty dictionary.

    Returns
    -------
    anndata.AnnData
        A concatenated AnnData object containing all sample embeddings, with patient and token metadata in `.obs`.

    Notes
    -----
    - Each observation's name is made unique using the associated sample ID as a prefix.
    - Observation metadata includes patient IDs, token IDs, token names, alternative token names (using the mapper), and values.
    - Variable metadata is labeled as E0, E1, ..., matching embedding dimensions.
    - Model is run in evaluation mode and attentions are not output.

    Example
    -------
    >>> emb_dataset = get_embeddings(my_model, my_data, my_sample_ids, my_tokenizer, device="cuda", mapper=my_feature_map)
    >>> emb_dataset.obs.head()
    """
    model.eval()
    adatas = []
    for ix in tqdm(range(len(data))):
        t,v,l = data.__getitem__(ix)
    
        v = v[t > 0].unsqueeze(0).to(device)
        t = t[t > 0].unsqueeze(0).to(device)
        
        out = model(tokens=t, values=v, output_last_states=True, output_attentions=False)
    
        sout = csr_matrix(out.last_hidden_state[0].cpu().detach().numpy())
        adata = ad.AnnData(sout)
        prefix = f"{sample_ids[ix]}_"
        adata.obs_names = [f"{prefix}{i}" for i in range(adata.n_obs)]
        adata.var_names = [f"E{i}" for i in range(adata.n_vars)]
        # adata.obs['PATIENT_ID'] = sample_ids[ix].astype("category")
        adata.obs['token_id'] = [int(i) for i in t[0]]
        adata.obs['token_name'] = [preprocessor.feature_decoder[int(i)] for i in t[0]]
        if mapper:
            adata.obs['token_name2'] = [mapper.get(preprocessor.feature_decoder[int(i)], preprocessor.feature_decoder[int(i)]) for i in t[0]]
        # adata.obs['value'] = v[0].cpu().detach().numpy()
        
        os.makedirs(f'{path}', exist_ok=True)
        adata.write_h5ad(f"{path}/{sample_ids[ix]}.h5ad", compression='gzip')