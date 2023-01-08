import pandas as pd


def read_benchmark_data(filename):
    foldername = "data/benchmark/"
    df = pd.read_csv(foldername + filename)
    data = df.values
    X, y = data[:, 0:-1], data[:, -1].astype(int)
    return X, y