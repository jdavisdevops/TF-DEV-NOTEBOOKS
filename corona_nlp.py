import tensorflow as tf
import pandas as pd
from pathlib import Path
import numpy as np
from pandasgui import show
cwd = Path.cwd()
data_dir = cwd / 'datasets'
train_csv = data_dir / 'Corona_NLP_train.csv'
test_csv = data_dir / 'Corona_NLP_test.csv'
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)


train_features = blue