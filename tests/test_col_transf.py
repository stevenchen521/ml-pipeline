import unittest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ..data_prep.col_transformer import DFColTransformer
from ..data_prep.func_transformer import LogTransformer
from sklearn.pipeline import Pipeline
from pandas import read_csv, concat, DataFrame
from pandas.core.series import Series
import numpy as np
from sklearn.model_selection import train_test_split

import os

MM_COLS = ['Highly Positive Skew']
STD_COLS = ['Moderate Positive Skew']
LOG_COLS = ['Moderate Negative Skew']
MM_LOG_COLS = ['Highly Negative Skew']
ROUND_SCALE = 8


def n_zero_diff(a, b, round_scale=ROUND_SCALE):
    a_inner = a
    if type(a) == DataFrame or type(a) == Series:
        a_inner = a.to_numpy()
    b_inner = b
    if type(b) == DataFrame or type(b) == Series:
        b_inner = b.to_numpy()

    diff = a_inner - b_inner
    diff_round = np.round(diff, round_scale)
    return np.count_nonzero(diff_round, axis=None)


class MyTestCase(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.df_test = read_csv(f'{current_dir}/data/data_to_transform.csv')
        cols = self.df_test.columns.to_numpy()
        cols = np.concatenate([["id"], cols])
        self.df_test["id"] = self.df_test.index + 1
        self.df_test = self.df_test[cols]

        print(f"Test data {self.df_test.columns.to_numpy()}")

    def test_mm_log_trans(self):
        df_test_inner = self.df_test.copy()
        mm_log_pipeline = Pipeline(steps=[('minMaxScaler', MinMaxScaler()), ('log_trans', LogTransformer())])
        def_res = mm_log_pipeline.fit_transform(df_test_inner[MM_LOG_COLS])

        test_mm = MinMaxScaler()
        test_log = LogTransformer()
        test_res_mm_log = test_mm.fit_transform(df_test_inner[MM_LOG_COLS])
        test_res_mm_log = test_log.fit_transform(test_res_mm_log)

        df_diff = test_res_mm_log - def_res
        diff_round = np.round(df_diff, ROUND_SCALE)
        assert np.count_nonzero(diff_round, axis=None) == 0

        # test the inverse transform
        test_res_inv = test_log.inverse_transform(test_res_mm_log)
        test_res_inv = test_mm.inverse_transform(test_res_inv)

        df_diff = df_test_inner[MM_LOG_COLS] - test_res_inv
        diff_round = np.round(df_diff, ROUND_SCALE)
        assert np.count_nonzero(diff_round, axis=None) == 0

    def test_col_fit_trans_invs(self):
        df_test_inner = self.df_test.copy()

        mm_log_pipeline = Pipeline(steps=[('minMaxScaler', MinMaxScaler()), ('log_trans', LogTransformer())])

        ct = DFColTransformer(
            [("minMax_scaler", MinMaxScaler(), MM_COLS),
             ("standard_scaler", StandardScaler(), STD_COLS),
             ("log_trans", LogTransformer(), LOG_COLS),
             ("mm_log_pipeline", mm_log_pipeline, MM_LOG_COLS)
             ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )
        trans_pipline = Pipeline(steps=[('DFColTransformer', ct)])
        df_res = trans_pipline.fit_transform(df_test_inner)

        # define test scalers
        test_mm_scalar = MinMaxScaler()
        test_std_scalar = StandardScaler()
        test_log = LogTransformer()

        mm_test_res = test_mm_scalar.fit_transform(df_test_inner[MM_COLS])
        assert n_zero_diff(df_res[MM_COLS], mm_test_res) == 0

        std_test_res = test_std_scalar.fit_transform(df_test_inner[STD_COLS])
        assert n_zero_diff(df_res[STD_COLS], std_test_res) == 0

        log_test_res = test_log.fit_transform(df_test_inner[LOG_COLS])
        assert n_zero_diff(df_res[LOG_COLS], log_test_res) == 0

        test_pl_mm_scalar = MinMaxScaler()
        test_pl_log_scalar = LogTransformer()
        test_pl_res_mmlog = test_pl_mm_scalar.fit_transform(df_test_inner[MM_LOG_COLS])
        test_pl_res_mmlog = test_pl_log_scalar.fit_transform(test_pl_res_mmlog)
        assert n_zero_diff(df_res[MM_LOG_COLS], test_pl_res_mmlog) == 0

        # test inverse transform
        df_res_inv = ct.inverse_transform(df_res)
        assert n_zero_diff(df_res_inv, df_res_inv) == 0

    def test_col_remainder_drop(self):
        """
        test parameter remainder="drop"
        :return:
        """
        df_test_inner = self.df_test.copy()

        mm_log_pipeline = Pipeline(steps=[('minMaxScaler', MinMaxScaler()), ('log_trans', LogTransformer())])
        ct = DFColTransformer(
            [("minMax_scaler", MinMaxScaler(), MM_COLS),
             ("log_trans", LogTransformer(), LOG_COLS),
             ("mm_log_pipeline", mm_log_pipeline, MM_LOG_COLS)
             ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        trans_pipline = Pipeline(steps=[('DFColTransformer', ct)])
        df_res = trans_pipline.fit_transform(df_test_inner)

        # 1, check column names matched
        target_cols = np.asarray([MM_COLS, LOG_COLS, MM_LOG_COLS]).reshape(1, -1).flatten()
        res_cols = df_res.columns.to_numpy()
        assert np.all(target_cols == res_cols)

        # 2. test inverse transform
        df_res_inv = ct.inverse_transform(concat([df_res, df_test_inner[STD_COLS]], axis=1))
        # df_res_inv = df_res_inv[df_test_inner.columns]
        assert n_zero_diff(df_test_inner[df_res_inv.columns], df_res_inv) == 0

    def test_col_verbose_feature_names_true(self):
        """
        Test verbose_feature_names_out = True
        :return:
        """
        df_test_inner = self.df_test.copy()
        ct = DFColTransformer(
            [
                ("mm_log_pipeline", Pipeline(steps=[('minMaxScaler', MinMaxScaler()), ('log_trans', LogTransformer())]),
                 MM_LOG_COLS),
                ("minMax_scaler", MinMaxScaler(), MM_COLS),
                ("log_trans", LogTransformer(), LOG_COLS),
            ],
            # remainder="drop",
            remainder="passthrough",
            verbose_feature_names_out=True
        )

        trans_pipline = Pipeline(steps=[('DFColTransformer', ct)])
        df_res = trans_pipline.fit_transform(df_test_inner)

        df_res_inv = trans_pipline.inverse_transform(df_res)
        # the column names contain transformer name, we rename them
        cols_t = []
        for col in df_res_inv.columns:
            cols_t.append(col.split("__")[1])
        df_res_inv.set_axis(cols_t, axis=1, inplace=True)
        # df_res_inv = df_res_inv[df_test_inner.columns]

        assert n_zero_diff(df_res_inv[cols_t], df_test_inner[cols_t]) == 0

    def test_col_transformers(self):
        df_test_inner = self.df_test.copy()
        ct = DFColTransformer(
            [
                ("minMax_scaler", MinMaxScaler(), None),
                ("mm_log_pipeline", Pipeline(steps=[('minMaxScaler', MinMaxScaler()), ('log_trans', LogTransformer())]),
                 MM_LOG_COLS),
            ],
            remainder="passthrough",
            verbose_feature_names_out=True
        )
        trans_pipline = Pipeline(steps=[('DFColTransformer', ct)])
        df_res = trans_pipline.fit_transform(df_test_inner)
        df_res_inv = trans_pipline.inverse_transform(df_res)

        cols_t = []
        for col in df_res_inv.columns:
            cols_t.append(col.split("__")[1])
        df_res_inv.set_axis(cols_t, axis=1, inplace=True)

        assert n_zero_diff(df_res_inv[cols_t], df_test_inner[cols_t]) == 0

    def test_train_test_sets(self):
        df_inner = self.df_test.copy()
        df_train, df_test = train_test_split(df_inner, test_size=0.3, shuffle=False)
        print(f"train set shape is {df_train.shape}")
        print(f"test set shape is {df_test.shape}")

        ct = DFColTransformer(
            [
                ("minMax_scaler", MinMaxScaler(), None),
                ("", None, ["id"]),
                ("mm_log_pipeline", Pipeline(steps=[('minMaxScaler', MinMaxScaler()), ('log_trans', LogTransformer())]),
                 MM_LOG_COLS),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )

        trans_pipline = Pipeline(steps=[('DFColTransformer', ct)])
        trans_pipline.fit(df_train)
        df_train_res = trans_pipline.transform(df_train)
        df_res_res = trans_pipline.transform(df_test)

        df_train_res_inv = trans_pipline.inverse_transform(df_train_res)
        df_test_res_inv = trans_pipline.inverse_transform(df_res_res)

        assert n_zero_diff(df_train_res_inv, df_train) == 0
        assert n_zero_diff(df_test_res_inv, df_test) == 0

    def test_same(self):
        df_inner = self.df_test.copy()
        df_train, df_test = train_test_split(df_inner, test_size=0.3, shuffle=False)
        print(f"train set shape is {df_train.shape}")
        print(f"test set shape is {df_test.shape}")

        ct = DFColTransformer(
            [
                ("minMax_scaler", MinMaxScaler(), None),
                ("", None, "id"),
                ("mm_log_pipeline", Pipeline(steps=[('minMaxScaler', MinMaxScaler()), ('log_trans', LogTransformer())]),
                 MM_LOG_COLS),
            ],
            remainder="passthrough"
        )

        trans_pipline = Pipeline(steps=[('DFColTransformer', ct)])
        trans_pipline.fit(df_train)
        df_train_res = trans_pipline.transform(df_train)
        df_res_res = trans_pipline.transform(df_test)

        df_train_res_inv = trans_pipline.inverse_transform(df_train_res)
        df_test_res_inv = trans_pipline.inverse_transform(df_res_res)

        assert n_zero_diff(df_train_res_inv["id"], df_train["id"]) == 0
        assert n_zero_diff(df_test_res_inv["id"], df_test["id"]) == 0


if __name__ == '__main__':
    unittest.main()
