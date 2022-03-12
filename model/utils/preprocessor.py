import pandas as pd
import logging

from numpy import float64


def get_data(filename):
    benign_df = pd.read_csv(filename)
    col_names = benign_df.columns.values
    for item in col_names:
        col_name = item.strip()
        benign_df.rename(columns={item: col_name}, inplace=True)
        if col_name in ['src', 'dst']:
            continue
        benign_df[col_name] = benign_df[col_name].replace(' None', '0')
        benign_df[col_name] = benign_df[col_name].astype(float64)
    logging.info('load {} records from file {}'.format(len(benign_df), filename))
    return benign_df
