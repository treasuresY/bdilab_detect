# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_data_bike_share(foldername="data/bike_share/"):
    df_train = pd.read_csv(foldername + "train.csv", parse_dates=['datetime'])
    df_test = pd.read_csv(foldername + "test.csv", parse_dates=['datetime'])
    df_train = df_train.drop(columns=['casual', 'registered'])
    full = pd.concat([df_train, df_test])
    full.sort_values(by='datetime', inplace=True)
    # full['date'] = full['datetime'].dt.date
    full['month'] = full['datetime'].dt.month
    full['hour'] = full['datetime'].dt.hour
    full['dayname'] = full['datetime'].dt.weekday
    full['year'] = full['datetime'].dt.year
    full.loc[full.weather == 4, 'weather'] = 3
    peak = [8, 16, 17, 18, 19]
    low = [22, 23, 0, 1, 2, 3, 4, 5, 6]
    full['hour'] = full['hour'].apply(lambda x: 3 if x in peak else (1 if x in low else 2))
    full = full.drop(columns=['holiday', 'atemp', 'datetime'])
    full = full.reset_index(drop=True)
    #y = full['count'].valuse()
    # X = full.values
    return full


def read_data_electricity_market(foldername="data/"):
    df = pd.read_csv(foldername + "elecNormNew.csv")
    data = df.values
    X, y = data[:, 1:-1], data[:, -1]

    # Set x,y as numeric
    X = X.astype(float)
    label = ["UP", "DOWN"]
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(y)

    return X, y

def read_data_weather(foldername="data/weather/"):
    df_labels = pd.read_csv(foldername + "NEweather_class.csv")
    y = df_labels.values.flatten()

    df_data = pd.read_csv(foldername + "NEweather_data.csv")
    X = df_data.values

    return X, y 


def read_data_forest_cover_type(foldername="data/"):
    df = pd.read_csv(foldername + "forestCoverType.csv")
    data = df.values
    X, y = data[:, 1:-1], data[:, -1]

    return X, y


def read_data_phishing(foldername="data/"):
    df = pd.read_csv(foldername + "phishing.csv")
    data = df.values
    X, y = data[:, 1:-1], data[:, -1]

    return X, y
