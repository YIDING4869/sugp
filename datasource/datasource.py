import numpy as np
import pandas as pd


class DataSource:  # accept DataFrame, drop probability, and seed
    def __init__(self, df, drop_probability=None, seed=123041):
        self.random = np.random.RandomState(seed)
        if drop_probability is not None:  # check_drop
            true_position = df[df['label'] == 1]  # get all true position
            remove_number = int(len(true_position) * drop_probability)  # find those label equal to true (1)
            drop_indexes = self.random.choice(true_position.index, remove_number, replace=False)
            # random remove n indices
            df = df.drop(drop_indexes).reset_index(drop=True)  # delete from dataframe。
            df.id = df.index  # update id

        self.df_index_id = df.set_index(["id"])  # set index to be the id
        self.df_sorted = df.sort_values(["proxy_score"], axis=0, ascending=False).reset_index(drop=True)
        # use the drop parameter to avoid the old index being added as a column:

    def get_label(self, ids) -> np.ndarray:
        return self.df_index_id.loc[ids]["label"].values

    def get_score(self, ids) -> np.ndarray:
        return self.df_index_id.loc[ids]['proxy_score'].values

    def get_sorted_id(self) -> np.ndarray:
        return self.df_sorted["id"].values

    def get_sorted_score(self) -> np.ndarray:
        return self.df_sorted["proxy_score"].values

    def get_true(self, ids) -> np.ndarray:
        labels = self.get_label(ids)
        return np.array([ids[i] for i in range(len(ids)) if labels[i]])


# get the csv source
def get_imagenet() -> DataSource:
    return get_csv_source('../data/csv/imagenet.csv')


def get_ontonotes() -> DataSource:
    return get_csv_source('../data/csv/ontonotes.csv')


def get_tacred() -> DataSource:
    return get_csv_source('../data/csv/tacred.csv')


def get_csv_source(file_path) -> DataSource:
    df = pd.read_csv(file_path)
    df['label'] = df['label'].astype('float32')  # save memory
    return DataSource(df)


def get_jackson_source(file_path, drop_probability=0.9, seed=1) -> DataSource:
    df = pd.read_feather(file_path)
    return DataSource(df, drop_probability == drop_probability, seed=seed)


def get_nightstreet() -> DataSource:
    # return get_jackson_source('../data/jackson/nightstreet.feather')
    return get_jackson_source('../data/jackson/2017-12-17.feather')

nightstreet_data = get_nightstreet()
print(f"Nightstreet data shape: \n {nightstreet_data.df_sorted}")

#data has already been read