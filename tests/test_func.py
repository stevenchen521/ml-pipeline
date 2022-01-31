import pytest
from data_prep.func_transformer import FuncTransformer, LogTransformer
from sklearn.pipeline import Pipeline
from pandas import DataFrame, read_csv
import os
import numpy as np

currentDir = os.path.dirname(__file__)


def test_log10():
   print(__name__)
   print(__package__)

   log_transformer = LogTransformer()
   pipline2 = Pipeline(steps=[('logtransformer', log_transformer)])

   df = read_csv(f'{currentDir}/data/data_to_transform.csv')

   df_log10 = pipline2.fit_transform(df)
   df_log10_inv = pipline2.inverse_transform(df_log10)

   df_diff = df_log10_inv - df

   diff_round = np.round(df_diff, 5)

   assert np.count_nonzero(diff_round, axis=None) == 0
