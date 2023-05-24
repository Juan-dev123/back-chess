from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Dropper(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(
            ['id', 'rated', 'created_at', 'last_move_at', 'victory_status', 'winner', 'increment_code', 'white_id',
             'white_rating', 'black_id', 'black_rating', 'moves', 'opening_eco', 'opening_name', 'opening_ply'], axis=1)


class PredictionStandardScaler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        stand_sca = StandardScaler()

        turns = stand_sca.fit_transform(X['turns'].values.reshape(-1, 1))
        X['turns'] = turns

        return X


class Preprocessing:
    final_model = Pipeline([
        ('dropper', Dropper()),
        ('standard_scaler', PredictionStandardScaler())
    ])
