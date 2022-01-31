from pandas import DataFrame
from sklearn.compose import ColumnTransformer


def convert_transformers(transformers, columns):
    cols_t = []
    default_idx = None
    for idx, transformer in enumerate(transformers):
        if transformer[2] is not None:
            cols_t.extend(transformer[2])
        else:
            default_idx = idx
    if default_idx is None:
        return transformers

    columns_list = columns.to_list()
    for col in cols_t:
        columns_list.remove(col)
    default_transformer = list(transformers[default_idx])
    default_transformer[2] = columns_list
    transformers[default_idx] = tuple(default_transformer)
    return transformers


class DFColTransformer(ColumnTransformer):

    def __init__(
            self,
            transformers,
            *,
            remainder="passthrough",
            sparse_threshold=0.3,
            n_jobs=None,
            transformer_weights=None,
            verbose=False,
            verbose_feature_names_out=False,
    ):
        super(DFColTransformer, self).__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out)
        self.features = None

    def fit(self, X, y=None):
        if type(X) != DataFrame:
            raise TypeError("Only DataFrame is accepted")
        self.transformers = convert_transformers(self.transformers, X.columns)
        self.features = X.columns
        return super().fit(X, y=None)

    def transform(self, X):
        if type(X) != DataFrame:
            raise TypeError("Only DataFrame is accepted")
        self.transformers = convert_transformers(self.transformers, X.columns)
        if type(X) == DataFrame:
            return DataFrame(super().transform(X), columns=self.features)[self.features]
        else:
            return super().transform(X)

    def fit_transform(self, X, y=None):
        if type(X) != DataFrame:
            raise TypeError("Only DataFrame is accepted")
        self.transformers = convert_transformers(self.transformers, X.columns)
        fit_res = super().fit_transform(X, y)
        if type(X) == DataFrame:
            self.features = X.columns
            if self.remainder == "passthrough":
                if self.get_feature_names_out() is not None:
                    if not self.verbose_feature_names_out:
                        return DataFrame(fit_res, columns=self.get_feature_names_out())[self.features]
                    else:
                        return DataFrame(fit_res, columns=self.get_feature_names_out())
                else:
                    return DataFrame(fit_res)
            elif self.remainder == "drop":
                return DataFrame(fit_res, columns=self.get_feature_names_out())
        else:
            return fit_res

    def inverse_transform(self, Xt):

        res_inner = Xt.copy()

        for transformer in self.transformers_:
            trans_inner_id = transformer[0]
            trans_inner = transformer[1]
            cols_t = transformer[2]
            if type(Xt) == DataFrame:
                if hasattr(trans_inner, 'inverse_transform'):
                    # if self.remainder == "drop" and self.verbose_feature_names_out == True:
                    if self.verbose_feature_names_out:
                        cols_t = [trans_inner_id + "__" + col_t for col_t in cols_t]
                    res_inner.loc[:, cols_t] = trans_inner.inverse_transform(Xt[cols_t])
        if not self.verbose_feature_names_out:
            return res_inner[self.features]
        else:
            return res_inner
