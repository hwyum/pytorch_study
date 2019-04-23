import os
import pandas as pd
from sklearn.model_selection import train_test_split

# data load
data_root = './data/'
data_path = os.path.join(data_root, 'ratings.txt')
data_df = pd.read_table(data_path)[['document','label']]
data_df = data_df[~data_df.document.isna()] ## document NA 있는 경우 제외

# train-test split
train_df, tst_df = train_test_split(data_df, test_size=0.2)

# train-validation split
train_df, val_df = train_test_split(train_df, test_size=0.2)

# save train / validation / test data
train_df.to_csv(os.path.join(data_root, 'tr_ratings.txt'), index=False, sep='\t')
val_df.to_csv(os.path.join(data_root, 'val_ratings.txt'), index=False, sep='\t')
tst_df.to_csv(os.path.join(data_root, 'tst_ratings.txt'), index=False, sep='\t')

