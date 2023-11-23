import pandas as pd

import fastparquet

import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.base import TransformerMixin

from functools import wraps


VERBOSE = True


def full_path(path: str) -> str:
    path_to_project = r'../'
    return path_to_project + path 


def logme(text: str, verbose=False, newline=False):
    def decorator(func):
        wraps(func)
        def wrapper(*args, **kwargs):
            if verbose: print(text, end='\n' if newline else '')
            result = func(*args, **kwargs)
            if verbose and not newline: print()
            return result
        return wrapper
    return decorator


class Aggregator():
    def __str__(self):
        return f"{self.df_agg.__class__.__name__}{self.df_agg.shape}"
    
    
    # def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
    #     return self.transform(df)
    
   
    @staticmethod
    def max_loan_months(s: pd.Series) -> int:
        chunks = [[]]
        for val in s.values:
            if val not in (1, 2):
                chunks.append([])
            else:
                chunks[-1].append(val)
        return max(map(len, chunks))
    
    
    @staticmethod
    def frac_loan_months(df: pd.DataFrame) -> pd.DataFrame:
        total_pay_months = (df != 3).sum(axis=1)
        total_pay_months = total_pay_months.apply(lambda x: 24 if x == 0 else x)
        return ((df == 1).sum(axis=1) + (df == 2).sum(axis=1)) / total_pay_months
    
    
    @staticmethod    
    def credit_history_length(s: pd.Series, enc_paym: list) -> int:
        for col in reversed(enc_paym):
            if s[col] != 3: 
                return int(col.lstrip('enc_paym_'))
        return 0
    
    
    ## Flags
    @logme('Aggregate flags: ', verbose=VERBOSE)
    def __agg_flags(self):
        self.df_agg.loc[:,self.flags] = self.df[['id'] + self.flags].groupby('id').agg('sum')
        if VERBOSE: print('▮', end='')
        
        
    ## Categorial encode
    @logme('Categorial encoding: ', verbose=VERBOSE)
    def __agg_cat(self):
        for col in self.cat:

            res = pd.DataFrame({'id': self.df.id, col: self.df[col]}, index=self.df.index)

            ohe = OneHotEncoder(sparse_output=False, dtype='int8')

            res = res.join(
                pd.DataFrame(
                    ohe.fit_transform(res[[col]]),
                    index=res.index,
                    columns=ohe.get_feature_names_out()
                )
            )

            res = res.drop(columns=col).groupby('id').agg('sum')   
            self.df_agg = self.df_agg.join(res, on=self.df_agg.index, how='left')

            if VERBOSE: print('▮', end='')

    def __chunked_apply(self, chunk_size: int, feature_title: str, *args, **kwargs):
        """make new feature <feature_title>, applying pd.DataFrame.apply() with *args/**kwargs chunk by chunk"""
        n = 0
        while n < self.df.shape[0]:
            n_init = n
            n = min(n + chunk_size, self.df.shape[0])
            chunk = self.df[self.enc_paym].iloc[n_init:n]\
                    .apply(*args, **kwargs).astype('int')
            self.df.loc[n_init:n, feature_title] = chunk
            if VERBOSE: print(n // chunk_size, end='')
    
    
    ## F.eng from enc_paym_N
    @logme('Features from enc_paym_N: ', verbose=VERBOSE)
    def __new_from_enc_paym(self):
        for col in ['enc_paym_11', 'enc_paym_20', 'enc_paym_24']:
            self.df[col] = self.df[col].apply(lambda x: x - 1)
        if VERBOSE: print('▮', end='')

        self.df['num_loan_months'] = self.df[self.enc_paym].apply(__class__.max_loan_months, axis=1)
        if VERBOSE: print('▮', end='')
        
        self.df['frac_loan_months'] = __class__.frac_loan_months(self.df[self.enc_paym])
        if VERBOSE: print('▮', end='')

        self.__chunked_apply(3_000_000, 'credit_history_length',
                             __class__.credit_history_length,
                             args=(self.enc_paym, ),
                             axis=1)
        if VERBOSE: print('▮', end='')

    
    ## Aggregate numerical -> max
    @logme('Aggregate numerical ', verbose=VERBOSE)
    def __agg_num(self):
        self.df_agg[[f"{col}_max" for col in self.num]] = self.df[['id'] + self.num].groupby('id').agg('max')
        if VERBOSE: print('▮', end='')

        
    ## Has any loan
    @logme('Adding has_loans feature', verbose=VERBOSE, newline=True)
    def __add_hasloans(self):
        cols_flag = [col for col in self.df_agg.columns if col.startswith('is_zero_loans')]
        self.df_agg.loc[:, 'has_loans'] = (self.df_agg[cols_flag] == 0).any(axis=1).astype('int8')
    
    
    ## Relative OHencoded features
    @logme('Relative OH-encoded features', verbose=VERBOSE, newline=True)
    def __make_relative(self):
        coded = self.df_agg.drop(columns=['rn_max', 'credit_history_length_max', 'num_loan_months_max', 'frac_loan_months_max', 'has_loans']).columns
        df_coded = self.df_agg[coded].div(self.df_agg.rn_max, axis=0)
        self.df_agg[coded] = df_coded
        
        
    ## Adding missed columns
    def __add_missed(self):
        features = joblib.load('features.pkl')
        missed = list(set(features) - set(self.df_agg.columns))
        if missed:
            self.df_agg.loc[:, missed] = 0
            self.df_agg = self.df_agg[features]
        if VERBOSE: print(f"Added {len(missed)} missed columns")

    ## Transform data
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.df = df.copy()
        self.df_agg = pd.DataFrame(index=df.id.unique())
    
        self.flags = [
            'is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90', 
            'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit', 'pclose_flag', 'fclose_flag', 
        ]
        self.enc_paym = [x for x in self.df.columns if x.startswith('enc_paym')]
        self.cat = [x for x in self.df.columns if x not in ['id', 'rn'] + self.flags + ['pre_loans6090', 'pre_over2limit', 'enc_loans_account_cur']]
        self.num = ['rn', 'credit_history_length', 'num_loan_months', 'frac_loan_months']
        
        self.__agg_flags()
        self.__agg_cat()
        self.__new_from_enc_paym()
        self.__agg_num()
        self.__add_hasloans()
        self.__make_relative()
        self.__add_missed()
        
        if VERBOSE: print(f"Aggregation completed. Result shape: {self.df_agg.shape}")
        
        return self.df_agg
    
    
    def fit(self, *args, **kwargs):
        return self