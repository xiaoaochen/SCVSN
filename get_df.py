import numpy as np
import pandas
from sklearn.model_selection import train_test_split


def get_df(vector_filename="dataset.pkl"):
    df = pandas.read_pickle(vector_filename)
    data = df
    vectors = np.stack(data.iloc[:, 0].values)
    labels = data.iloc[:, 1].values
    postitive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    undersampled_negative_idxs = np.random.choice(negative_idxs,
                                                  len(postitive_idxs),
                                                  replace=False)
    resampled_idexs = np.concatenate([postitive_idxs, undersampled_negative_idxs])
    x_train, x_test, y_train, y_test = train_test_split(
        vectors[resampled_idexs],
        labels[resampled_idexs],
        test_size=0.2,
        stratify=labels[resampled_idexs])
    input_shape = (vectors.shape[1], vectors.shape[2])
    return x_train, x_test, y_train, y_test, input_shape, labels
