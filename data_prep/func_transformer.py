import numpy as np
from sklearn.preprocessing import FunctionTransformer
from numpy import log10, power, sign, float64, abs, log
from pandas import DataFrame
from sklearn.base import _OneToOneFeatureMixin
from math import e


class FuncTransformer(FunctionTransformer, _OneToOneFeatureMixin):

    def __init__(
            self,
            func=None,
            inverse_func=None,
            *,
            validate=True,
            accept_sparse=False,
            check_inverse=True,
            kw_args=None,
            inv_kw_args=None, ) -> None:

        super(FuncTransformer, self).__init__(
            func=func,
            inverse_func=inverse_func,
            validate=validate,
            accept_sparse=accept_sparse,
            check_inverse=check_inverse,
            kw_args=kw_args,
            inv_kw_args=inv_kw_args)

        self.features = None

    def fit(self, X, y=None):
        if type(X) == DataFrame:
            self.features = X.columns.to_numpy()
        return super().fit(X, y)

    def transform(self, X):
        x_inner = X
        if type(X) == np.ndarray and hasattr(self, "features"):
            x_inner = DataFrame(X, columns=self.features)
        return super().transform(x_inner)

    def fit_transform(self, X, y=None, **fit_params):
        if type(X) == DataFrame:
            self.features = X.columns.to_numpy()
        return super().fit_transform(X, y=y, **fit_params)

    def get_feature_names_out(self, input_features=None):
        return super().get_feature_names_out(input_features)

    def inverse_transform(self, X):
        if type(X) == DataFrame:
            self.features = X.columns.to_numpy()
        return super().inverse_transform(X)


def log_trans(X):
    return log(1 + abs(X)) * sign(X).astype(float64)


def log_trans_inv(X):
    return (power(e, abs(X)) - 1) * sign(X).astype(float64)


def log10_trans(X):
    return log10(1 + abs(X)) * sign(X).astype(float64)


def log10_trans_inv(X):
    return (power(10, abs(X)) - 1) * sign(X).astype(float64)


class LogTransformer(FuncTransformer):

    def __init__(
            self,
            func=log_trans,
            inverse_func=log_trans_inv,
            validate=True,
            accept_sparse=False,
            check_inverse=True,
            kw_args=None,
            inv_kw_args=None, ) -> None:
        super(FuncTransformer, self).__init__(
            func=func,
            inverse_func=inverse_func,
            validate=validate,
            accept_sparse=accept_sparse,
            check_inverse=check_inverse,
            kw_args=kw_args,
            inv_kw_args=inv_kw_args)
