import pandas as pd

import fastparquet

import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix


def full_path(path: str) -> str:
    path_to_project = r'C:\Users\Arseniy\Documents\Skillbox\diploma_ML/'
    return path_to_project + path 


def max_loan_months(s: pd.Series) -> int:
    chunks = [[]]
    for val in s.values:
        if val not in (1, 2):
            chunks.append([])
        else:
            chunks[-1].append(val)
    return max(map(len, chunks))


def frac_loan_months(df: pd.DataFrame) -> pd.DataFrame:
    total_pay_months = (df != 3).sum(axis=1)
    total_pay_months = total_pay_months.apply(lambda x: 24 if x == 0 else x)
    return ((df == 1).sum(axis=1) + (df == 2).sum(axis=1)) / total_pay_months


def credit_history_length(s: pd.Series) -> int:
    for col in reversed(s.index):
        if s[col] != 3: 
            return int(col.lstrip('enc_paym_'))
    return 0


def aggregate(df: pd.DataFrame, verbose=False) -> pd.DataFrame:    

    df_agg = pd.DataFrame(index=df.id.unique())
    
    ## Flags
    if verbose: print('Aggregate flags: ', end='')
    flags = [
        'is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90', 
        'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit', 'pclose_flag', 'fclose_flag', 
    ]
    df_agg.loc[:,flags] = df[['id'] + flags].groupby('id').agg('sum')
    if verbose: print('▮')
        
    ## Categorial encode
    if verbose: print('Categorial encoding: ', end='')
    
    cat = [x for x in df.columns if x not in ['id', 'rn'] + flags + ['pre_loans6090', 'pre_over2limit', 'enc_loans_account_cur']]
    
    for col in cat:

        res = pd.DataFrame({'id': df.id, col: df[col]}, index=df.index)

        ohe = OneHotEncoder(sparse_output=False, dtype='int8')
        # ohe.fit(res[[col]])

        res = res.join(
            pd.DataFrame(
                ohe.fit_transform(res[[col]]),
                index=res.index,
                columns=ohe.get_feature_names_out()
            )
        )
        # print(col, res.shape)

        res = res.drop(columns=col).groupby('id').agg('sum')   
        df_agg = df_agg.join(res, on=df_agg.index, how='left')
        
        if verbose: print('▮', end='')

    if verbose: print()
    
    ## F.eng from enc_paym_N
    if verbose: print('Feature eng. from enc_paym_N: ', end='')

    enc_paym = [x for x in df.columns if x.startswith('enc_paym')]
    for col in ['enc_paym_11', 'enc_paym_20', 'enc_paym_24']:
        df[col] = df[col].apply(lambda x: x - 1)
    if verbose: print('▮', end='')
    
    df['num_loan_months'] = df[enc_paym].apply(max_loan_months, axis=1)
    if verbose: print('▮', end='')
    df['frac_loan_months'] = frac_loan_months(df[enc_paym])
    if verbose: print('▮', end='')
    
    chunk_size = 3_000_000
    n = 0
    while n < df.shape[0]:
        n_init = n
        n = min(n + chunk_size, df.shape[0])
        chunk = df[enc_paym].iloc[n_init:n].apply(credit_history_length, axis=1).astype('int').copy()
        df.loc[n_init:n, 'credit_history_length'] = chunk
        if verbose: print(n // chunk_size, end='')
    if verbose: print('▮', end='')
    
    if verbose: print()
    
    ## Aggregate numerical -> max
    if verbose: print("Aggregate numerical ", end='')
    
    num = ['rn', 'credit_history_length', 'num_loan_months', 'frac_loan_months']
    df_agg[[f"{col}_max" for col in num]] = df[['id'] + num].groupby('id').agg('max')
    if verbose: print('▮', end='')
    
    if verbose: print()
    
    ## Adding missed columns
    features = joblib.load('features.pkl')
    missed = list(set(features) - set(df_agg.columns))
    if missed:
        df_agg.loc[:, missed] = 0
        df_agg = df_agg[features]
    if verbose: print(f"Added {len(missed)} missed columns")
    
    
    if verbose: print(f"Aggregation completed. Result shape: {df_agg.shape}")
    return df_agg