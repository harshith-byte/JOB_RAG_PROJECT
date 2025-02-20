import pandas as pd
import os
import re
def data_cleaning():
    if os.path.exists('data/updated_file.csv'):
        return pd.read_csv('data/updated_file.csv',on_bad_lines='skip',sep=',')
    
    df = pd.read_csv('data/jobs_en.csv',on_bad_lines='skip',sep='|')

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.drop('raw_jobs_crawl_id',axis='columns', inplace=True)

    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df['description'] = df['description'].apply(lambda x: x.strip())
    df.to_csv('data/updated_file.csv', index=False)

    return df