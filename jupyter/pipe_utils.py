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

    ## Categorial encode
    if verbose: print('Categorial encoding: ', end='')
    
    cat = [
        'pre_since_opened', 'pre_since_confirmed', 'pre_pterm', 'pre_fterm', 'pre_till_pclose', 'pre_till_fclose', 
        'pre_loans_credit_limit', 'pre_loans_next_pay_summ', 'pre_loans_outstanding', 
        'pre_loans_max_overdue_sum', 'pre_loans_credit_cost_rate', 
        'pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90', 
        'pre_util', 'pre_over2limit', 'pre_maxover2limit',  
        'enc_loans_account_holder_type', 'enc_loans_credit_status', 'enc_loans_account_cur', 'enc_loans_credit_type', 
        'pclose_flag', 'fclose_flag',
    ]
    
    for col in cat:

        res = pd.DataFrame({'id': df.id, col: df[col]}, index=df.index)

        ohe = OneHotEncoder(sparse_output=False, dtype='int8')

        res = res.join(
            pd.DataFrame(
                ohe.fit_transform(res[[col]]),
                index=res.index,
                columns=ohe.get_feature_names_out()
            )
        )

        res = res.drop(columns=col).groupby('id').agg('sum')   
        df_agg = df_agg.join(res, on=df_agg.index, how='left')
        
        if verbose: print('▮', end='')

    if verbose: print(' done')
        
    ## Flags
    if verbose: print('Feature eng. from is_zero_loans ', end='')
    
    cols_flag = [
        'is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90',  
    ]
    flags = [
        'has_loans', 'has_loans560', 'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit',
    ]

    df['has_loans'] = (df[cols_flag] == 0).any(axis=1).astype('int8')
    if verbose: print('▮', end='')
    # df['has_loans560'] = df.apply(lambda s: s.is_zero_loans530 == 0 and s.is_zero_loans3060 == 0, axis=1).astype('int8')
    df['has_loans560'] = ((df.is_zero_loans530 == 0) & (df.is_zero_loans3060 == 0)).astype('int8')
    if verbose: print('▮', end='')
    
    if verbose: print(' done')
    if verbose: print('Aggregate flags ▮ ', end='')

    res = df[['id'] + flags].groupby('id').agg(any).astype('int8') 
    df_agg = df_agg.join(res, on=df_agg.index, how='left')
    
    if verbose: print('done')

    ## F.eng from enc_paym_N
    if verbose: print('Feature eng. from enc_paym_N: ', end='')
    
    enc_paym = [x for x in df.columns if x.startswith('enc_paym')]
    for col in ['enc_paym_11', 'enc_paym_20', 'enc_paym_24']:
        df[col] = df[col].apply(lambda x: x - 1)
    if verbose: print('▮', end='')

    df['max_loan_months'] = df[enc_paym].apply(max_loan_months, axis=1)
    if verbose: print('▮', end='')
    df['frac_loan_months'] = frac_loan_months(df[enc_paym])
    if verbose: print('▮', end='')
    
    chunk_size = 2_000_000
    n = 0
    while n < df.shape[0]:
        n = min(n + chunk_size, df.shape[0])
        chunk = df[enc_paym].iloc[n-chunk_size:n].apply(credit_history_length, axis=1).astype('int8').copy()
        df.loc[n-chunk_size:n, 'credit_history_length'] = chunk
        if verbose: print(n // chunk_size, end='')
        
    if verbose: print('▮', end='')

    if verbose: print(' done')
    
    ## Numerical
    if verbose: print('Numerical aggregation ▮ ', end='')
    
    num = [
        'rn', 'max_loan_months', 'frac_loan_months', 'credit_history_length',
    ]

    res = df[['id'] + num].groupby('id').agg('max')   
    df_agg = df_agg.join(res, on=df_agg.index, how='left')
    
    if verbose: print('done')
    
    if verbose: print(f"Aggregation completed. Result shape: {df_agg.shape}")
    
    return df_agg

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.copy()
    cat = [
        'pre_since_opened', 'pre_since_confirmed', 'pre_pterm', 'pre_fterm', 'pre_till_pclose', 'pre_till_fclose', 
        'pre_loans_credit_limit', 'pre_loans_next_pay_summ', 'pre_loans_outstanding', 
        'pre_loans_max_overdue_sum', 'pre_loans_credit_cost_rate', 
        'pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90', 
        'pre_util', 'pre_over2limit', 'pre_maxover2limit',  
        'enc_loans_account_holder_type', 'enc_loans_credit_status', 'enc_loans_account_cur', 'enc_loans_credit_type', 
        'pclose_flag', 'fclose_flag',
    ]
    num = [
        'rn', 'max_loan_months', 'frac_loan_months', 'credit_history_length',
    ]
    flags = [
        'has_loans', 'has_loans560', 'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit',
    ]
    cols = ['id'] + cat + num + flags
    
    ## relative one-hot encoded features
    features = [x for x in df.columns if x not in ('rn', 'target', 'frac_loan_months', 'has_loans', 'has_loans560', 'max_loan_months', 'credit_history_length',
                                              'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit',)]
    df[features] = df[features].apply(lambda x: x / df.rn)
   
    ## drop columns with a single value
    df.drop(columns=[
        'pre_loans530_11', 'pre_loans530_7', 'pre_loans530_19', 'pre_loans6090_3', 'pre_loans3060_1', 'pre_loans5_9', 'pre_loans5_8', 'pre_loans_max_overdue_sum_0', 'pre_loans3060_3', 'pre_loans5_11', 'pre_loans90_3', 'pre_loans3060_0', 'pre_loans6090_0', 'pre_loans3060_4', 'pre_loans3060_6', 'pre_loans5_10', 'pre_loans530_17', 'pre_loans530_8', 'pre_loans530_9', 'pre_loans530_5'
    ],  inplace=True, errors='ignore')
            
    ## standardization
    num = ['rn', 'max_loan_months', 'credit_history_length']
    ssc = StandardScaler()
    df = df.drop(columns=num).join(
        pd.DataFrame(
            ssc.fit_transform(df[num]),
            index=df.index,
            columns=[f"{x}_std" for x in num]
        )
    )
    
    # all_cols = joblib.load(full_path('model/features.pkl'))
    # lost_cols = list(set(all_cols) - set(df.columns))
    # if lost_cols:
    #     df[lost_cols] = 0
        
    return df