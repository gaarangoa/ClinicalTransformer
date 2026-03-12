# Build dataset
import pandas as pd
import numpy as np
import anndata
from scipy.sparse import coo_matrix
from clinical_transformer import vnBertTokenizerTabular as TokenizerTabular

model_id = 'MyModel'
DATA_PATH = f'../../data/{model_id}/NIHMS1819096-supplement-Supp__Table_3.xlsx'
TOKENIZER_SAVE_DIR = f'../../models/ssl/{model_id}/tokenizer/'
OUTPUT_PATH = f'../../data/{model_id}/training_data.h5ad'

df = pd.read_excel(DATA_PATH)
print(f'Loaded {df.shape[0]} samples x {df.shape[1]} columns')
df.head(3)

# Categorical features (string-valued columns)
categorical_features = [
    'Cancer_type_grouped_2',
    'Cancer_Type',
    'Stage at IO start',
    'Drug_class',
]

# Numerical features (continuous + binary-coded columns)
numerical_features = [
    'Age',
    'BMI',
    'NLR',
    'Platelets',
    'HGB',
    'Albumin',
    'TMB',
    'FCNA',
    'HED',
    'MSI_SCORE',
    'Chemo_before_IO (1:Yes; 0:No)',
    'Sex (1:Male; 0:Female)',
    'Stage (1:IV; 0:I-III)',
    'Drug (1:Combo; 0:PD1/PDL1orCTLA4)',
    'HLA_LOH',
    'MSI (1:Unstable; 0:Stable_Indeterminate)',
    'Cancer_Type2',
]

# Columns to keep as obs metadata (not tokenised)
label_columns = [
    'SAMPLE_ID',
    'Response (1:Responder; 0:Non-responder)',
    'OS_Event',
    'OS_Months',
    'PFS_Event',
    'PFS_Months',
    'RF16_prob',
]

print(f'Categorical: {len(categorical_features)}')
print(f'Numerical:   {len(numerical_features)}')
print(f'Labels/meta: {len(label_columns)}')


# Tokenizer
# Convert rows to list-of-dicts
feature_cols = categorical_features + numerical_features
samples = df[feature_cols].to_dict(orient='records')

tokenizer = TokenizerTabular()
tokenizer.fit(
    samples,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
)

tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)

# encode samples 
encoded = tokenizer(
    samples,
    return_tensors=None,
    return_attention_mask=False,
    return_minmax_values=True,
)

# Pick sample 0 as example
sample_raw = samples[250]
sample_enc = tokenizer.encode_sample(sample_raw, return_minmax_values=True)

rows_example = []
for i, tid in enumerate(sample_enc['input_ids']):
    feat = tokenizer.convert_ids_to_tokens(tid)
    rows_example.append({
        'feature': feat,
        'token_id': tid,
        'raw_value': sample_raw.get(feat, None),
        'resolved_numeric': sample_enc['raw_values'][i],
        'minmax_encoded': round(sample_enc['minmax_values'][i], 4),
    })

example_df = pd.DataFrame(rows_example)
example_df


# Build sparse matrix Ann data
cols = encoded['input_ids']
values = encoded['minmax_values']

rows = []
for ix, v in enumerate(values):
    rows += [ix] * len(v)

cols_flat = np.concatenate(cols)
values_flat = np.concatenate(values)
rows_flat = np.array(rows)

print(f'rows: {rows_flat.shape}, cols: {cols_flat.shape}, values: {values_flat.shape}')
sparse_matrix = coo_matrix(
    (values_flat, (rows_flat, cols_flat)),
    shape=(len(samples), tokenizer.vocab_size),
)

adata = anndata.AnnData(X=sparse_matrix.tocsr())

# Attach sample IDs and labels as obs metadata
adata.obs.index = df['SAMPLE_ID'].astype(str).values
for col in label_columns:
    adata.obs[col] = df[col].values

print(adata)
adata.obs.head()

# Save to disk
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
adata.write(OUTPUT_PATH)
print(f'Saved to {OUTPUT_PATH}')
print(f'Shape: {adata.shape}')

# Reload and sanity-check
adata2 = anndata.read_h5ad(OUTPUT_PATH)
print(f'Reloaded shape: {adata2.shape}')
print(f'Non-zero entries (sample 0): {np.sum(adata2[0].X.toarray() != 0)}')
print(f'Obs columns: {list(adata2.obs.columns)}')
adata2

# move it to shared shm 
# cp ../../data/CT_chowell_pretrained_v2.0/training_data.h5ad /dev/shm/training_data.h5ad





