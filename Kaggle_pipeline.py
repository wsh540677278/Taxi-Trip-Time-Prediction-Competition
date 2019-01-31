import numpy as np
import time
import sys
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from math import log, exp, sqrt
import math
from geopy.distance import vincenty
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, date

from sklearn import tree
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

# ----------------------------------------------------helper function -------------------------------------


def timestamp_datetime(value):
    date_format = '%Y|%m|%d|%H|%M|%S'
    value = time.localtime(value)
    dt = time.strftime(date_format, value)
    return dt


def is_go_through(ts, duration, tp):
    current = timestamp_datetime(ts)
    current_day = current[:11]
    snapshot = current_day + tp
    return ts + duration >= datetime_timestamp(snapshot) >= ts


def get_second_to_shapshot(ts, tp):
    current = timestamp_datetime(ts)
    current_day = current[:11]
    snapshot = current_day + tp
    return datetime_timestamp(snapshot) - ts


def datetime_timestamp(dt):
    time.strptime(dt, '%Y|%m|%d|%H|%M|%S')
    s = time.mktime(time.strptime(dt, '%Y|%m|%d|%H|%M|%S'))
    return int(s)


def get_weekday(value):
    value = time.localtime(value)
    return time.strftime("%w", value)


def get_sh(df):
    time_series = []
    for i in tqdm(range(len(df))):
        ts = timestamp_datetime(df.iloc[i].TIMESTAMP)
        time_series.append(ts[11:13])
    df['SH'] = time_series


def cv(model, X, Y):
    scores = cross_val_score(model, X, Y, cv=5, n_jobs=5, scoring='neg_mean_squared_error')
    print(math.sqrt(abs(scores.mean())))


def norm(X):
    return MinMaxScaler().fit_transform(X)


def draw_pic(model, X, Y):
    model.fit(X, Y)
    result = model.predict(X)
    plt.figure()
    plt.plot(np.arange(len(result)), Y, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % mean_squared_error(Y, result))
    plt.legend()
    plt.show()


def submit2(predicts, index, name):
    index = index.reshape(len(index), 1)
    predicts = predicts.reshape(len(predicts), 1)
    new_np = np.concatenate((index, predicts), axis=1)
    new_df = pd.DataFrame(new_np)
    # new_df.reset_index()
    new_df.columns = ['TRIP_ID', 'TRAVEL_TIME']
    # new_df.to_csv(name, index=False)
    return new_df

# ----------------------------------------------------------------------------------------------------------


def feature_engineering(df1, df3):

    def one_hot(df, string):
        return pd.get_dummies(df, prefix=string)

    df1['WEEKDAY'] = df1.TIMESTAMP.apply(get_weekday)
    weekday_df = one_hot(df1['WEEKDAY'], 'WEEKDAY')
    df1 = pd.concat([df1, weekday_df], axis=1)

    df3['WEEKDAY'] = df3.TIMESTAMP.apply(get_weekday)
    test_weekday_df = one_hot(df3['WEEKDAY'], 'WEEKDAY')
    df3 = pd.concat([df3, test_weekday_df], axis=1)

    def get_datt(df, test_df):
        speed_map = {}
        for i in tqdm(range(len(df))):
            if df.iloc[i].TAXI_ID in speed_map:
                speed_map[df.iloc[i].TAXI_ID][0] += df.iloc[i].TRIP_TIME
                speed_map[df.iloc[i].TAXI_ID][1] += 1
            else:
                speed_map[df.iloc[i].TAXI_ID] = [df.iloc[i].TRIP_TIME, 1]
        datt_list = []
        for i in tqdm(range(len(df))):
            datt_list.append(speed_map[df.iloc[i].TAXI_ID][0] / speed_map[df.iloc[i].TAXI_ID][1])
        test_datt_list = []
        for i in tqdm(range(len(test_df))):
            if test_df.iloc[i].TAXI_ID in speed_map:
                test_datt_list.append(speed_map[test_df.iloc[i].TAXI_ID][0] / speed_map[test_df.iloc[i].TAXI_ID][1])
            else:
                test_datt_list.append(0.0)
        df['DATT'] = datt_list
        test_df['DATT'] = test_datt_list
        return

    get_datt(df1, df3)

    def get_sp(df, test_df):
        stand_popularity_map = {}
        for i in tqdm(range(len(df))):
            if df.iloc[i].CALL_TYPE == 'B' and df.iloc[i].ORIGIN_STAND > 0:
                if df.iloc[i].ORIGIN_STAND in stand_popularity_map:
                    stand_popularity_map[df.iloc[i].ORIGIN_STAND] += 1
                else:
                    stand_popularity_map[df.iloc[i].ORIGIN_STAND] = 1
        sp_list = []
        for i in tqdm(range(len(df))):
            if df.iloc[i].CALL_TYPE == 'B' and df.iloc[i].ORIGIN_STAND > 0:
                sp_list.append(stand_popularity_map[df.iloc[i].ORIGIN_STAND])
            else:
                sp_list.append(0.0)
        df['SP'] = sp_list

        test_sp_list = []
        for i in tqdm(range(len(test_df))):
            if test_df.iloc[i].CALL_TYPE == 'B' and test_df.iloc[i].ORIGIN_STAND > 0:
                test_sp_list.append(stand_popularity_map[test_df.iloc[i].ORIGIN_STAND])
            else:
                test_sp_list.append(0.0)
        test_df['SP'] = test_sp_list

    def get_catt(df, test_df):
        customer_map = {}
        for i in tqdm(range(len(df))):
            if df.iloc[i].CALL_TYPE == 'A':
                if df.iloc[i].ORIGIN_CALL in customer_map:
                    customer_map[df.iloc[i].ORIGIN_CALL][0] += df.iloc[i].TRIP_TIME
                    customer_map[df.iloc[i].ORIGIN_CALL][1] += 1
                else:
                    customer_map[df.iloc[i].ORIGIN_CALL] = [df.iloc[i].TRIP_TIME, 1]
        catt_list = []
        for i in tqdm(range(len(df))):
            if df.iloc[i].CALL_TYPE == 'A':
                catt_val = customer_map[df.iloc[i].ORIGIN_CALL][0] / customer_map[df.iloc[i].ORIGIN_CALL][1]
                catt_list.append(catt_val)
            else:
                catt_list.append(np.nan)
        df['CATT'] = catt_list

        test_catt_list = []
        for i in tqdm(range(len(test_df))):
            if test_df.iloc[i].CALL_TYPE == 'A' and test_df.iloc[i].ORIGIN_CALL in customer_map:
                catt_val = customer_map[test_df.iloc[i].ORIGIN_CALL][0] / customer_map[test_df.iloc[i].ORIGIN_CALL][1]
                test_catt_list.append(catt_val)
            else:
                test_catt_list.append(0.0)
        test_df['CATT'] = test_catt_list

    get_sp(df1, df3)
    get_catt(df1, df3)

    df1.to_csv("df1.csv")
    df3.to_csv('df3.csv')
    return df1, df3


def train_model_a(df1, df3):
    df1 = df1.loc[(df1.MISSING_DATA == False) & (df1.CALL_TYPE == 'A')]
    print("the shape of type A data", df1.shape)
    # patterns = ["2014|08|14|", "2014|09|30|", "2014|10|06|", "2014|11|1|", "2014|12|21|"]
    # weekday_patterns = ["4", "2", "1", "5", "0"]
    time_patterns = ['14|00|00', '04|30|00', '13|45|00', '23|59|59', '09|30|00']
    result = []
    sts_list = []
    for i in tqdm(range(len(df1))):
        for j in time_patterns:
            if is_go_through(df1.iloc[i].TIMESTAMP, df1.iloc[i].TRIP_TIME, j):
                result.append(df1.iloc[i])
                sts_list.append(get_second_to_shapshot(df1.iloc[i].TIMESTAMP, j))
                break
    df1 = pd.DataFrame(result)
    df1['STS'] = sts_list

    df3 = df3.loc[(df3.MISSING_DATA == False) & (df3.CALL_TYPE == 'A')]
    print("the shape of type A data", df3.shape)
    # patterns = ["2014|08|14|", "2014|09|30|", "2014|10|06|", "2014|11|1|", "2014|12|21|"]
    weekday_patterns = ["4", "2", "1", "5", "0"]
    time_patterns = ['14|00|00', '04|30|00', '13|45|00', '23|59|59', '09|30|00']
    result = []
    sts_list = []
    for i in tqdm(range(len(df3))):
        weekday = get_weekday(df3.iloc[i].TIMESTAMP)
        if weekday in weekday_patterns:
            index_ = weekday_patterns.index(weekday)
            result.append(df3.iloc[i])
            sts_list.append(get_second_to_shapshot(df3.iloc[i].TIMESTAMP, time_patterns[index_]))
    df3 = pd.DataFrame(result)
    df3['STS'] = sts_list

    get_sh(df1)
    get_sh(df3)

    def build_train_data_a(df):
        new_df = df.loc[:,
                 ['DATT', 'TIMESTAMP', 'SH', 'STS', 'TRIP_TIME', 'ORIGIN_CALL', 'CATT', 'WEEKDAY_0', 'WEEKDAY_1',
                  'WEEKDAY_2', 'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5', 'WEEKDAY_6']]
        new_df = new_df.dropna(axis=0, how='any')
        train_Y = np.array(new_df['TRIP_TIME'])
        train_X = np.array(new_df.loc[:, ['DATT', 'TIMESTAMP', 'STS', 'SH', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                                          'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5', 'WEEKDAY_6']])
        print("The shape of training data", train_X.shape)
        return train_X, train_Y

    def build_test_data_a(df):
        new_df = df.loc[:, ['TRIP_ID', 'DATT', 'TIMESTAMP', 'SH', 'STS', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                            'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5', 'WEEKDAY_6']]
        new_df = new_df.dropna(axis=0, how='any')
        indexes = np.array(new_df['TRIP_ID'])
        test_X = np.array(new_df.loc[:, ['DATT', 'TIMESTAMP', 'SH', 'STS', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                                         'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5', 'WEEKDAY_6']])
        print("The shape of test data", test_X.shape)
        return test_X, indexes

    X, Y = build_train_data_a(df1)
    test_X = build_test_data_a(df3)
    model = XGBRegressor()
    model.fit(X, Y)
    predicts = model.predict(test_X[0])
    res = submit2(predicts, test_X[1], "submission_a.csv")
    return res


def train_model_c(df1, df3):
    df1 = df1.loc[(df1.MISSING_DATA == False) & (df1.CALL_TYPE == 'C')]
    print("the shape of type C data", df1.shape)
    # patterns = ["2014|08|14|", "2014|09|30|", "2014|10|06|", "2014|11|1|", "2014|12|21|"]
    # weekday_patterns = ["4", "2", "1", "5", "0"]
    time_patterns = ['14|00|00', '04|30|00', '13|45|00', '23|59|59', '09|30|00']
    result = []
    sts_list = []
    for i in tqdm(range(len(df1))):
        for j in time_patterns:
            if is_go_through(df1.iloc[i].TIMESTAMP, df1.iloc[i].TRIP_TIME, j):
                result.append(df1.iloc[i])
                sts_list.append(get_second_to_shapshot(df1.iloc[i].TIMESTAMP, j))
                break
    df1 = pd.DataFrame(result)
    df1['STS'] = sts_list

    df3 = df3.loc[(df3.MISSING_DATA == False) & (df3.CALL_TYPE == 'C')]
    print("the shape of type C data", df3.shape)
    # patterns = ["2014|08|14|", "2014|09|30|", "2014|10|06|", "2014|11|1|", "2014|12|21|"]
    weekday_patterns = ["4", "2", "1", "5", "0"]
    time_patterns = ['14|00|00', '04|30|00', '13|45|00', '23|59|59', '09|30|00']
    result = []
    sts_list = []
    for i in tqdm(range(len(df3))):
        weekday = get_weekday(df3.iloc[i].TIMESTAMP)
        if weekday in weekday_patterns:
            index = weekday_patterns.index(weekday)
            result.append(df3.iloc[i])
            sts_list.append(get_second_to_shapshot(df3.iloc[i].TIMESTAMP, time_patterns[index]))
    df3 = pd.DataFrame(result)
    df3['STS'] = sts_list

    get_sh(df1)
    get_sh(df3)

    def build_train_data_c(df):
        new_df = df.loc[:, ['DATT', 'TIMESTAMP', 'SH', 'STS', 'TRIP_TIME', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                            'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5', 'WEEKDAY_6']]
        new_df = new_df.dropna(axis=0, how='any')
        train_Y = np.array(new_df['TRIP_TIME'])
        train_X = np.array(new_df.loc[:, ['DATT', 'TIMESTAMP', 'STS', 'SH', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                                          'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5', 'WEEKDAY_6']])
        print("The shape of training data", train_X.shape)
        return train_X, train_Y

    def build_test_data_c(df):
        new_df = df.loc[:, ['TRIP_ID', 'DATT', 'TIMESTAMP', 'SH', 'STS', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                            'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5', 'WEEKDAY_6']]
        new_df = new_df.dropna(axis=0, how='any')
        indexes = np.array(new_df['TRIP_ID'])
        test_X = np.array(new_df.loc[:, ['DATT', 'TIMESTAMP', 'SH', 'STS', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                                         'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5', 'WEEKDAY_6']])
        print("The shape of test data", test_X.shape)
        return test_X, indexes

    X, Y = build_train_data_c(df1)
    test_X = build_test_data_c(df3)
    model = XGBRegressor()
    model.fit(X, Y)
    predicts = model.predict(test_X[0])
    submit2(predicts, test_X[1], "submission_c.csv")
    res = df3.loc[:, ['TRIP_ID', 'STCT']]
    res.columns = ['TRIP_ID', 'TRAVEL_TIME']
    return res


def train_model_b(df1, df3, df4):
    df1 = df1.loc[(df1.MISSING_DATA == False) & (df1.CALL_TYPE == 'B') & (df1.ORIGIN_STAND > 0)]
    print("the shape of type B data", df1.shape)
    # patterns = ["2014|08|14|", "2014|09|30|", "2014|10|06|", "2014|11|1|", "2014|12|21|"]
    # weekday_patterns = ["4", "2", "1", "5", "0"]
    time_patterns = ['14|00|00', '04|30|00', '13|45|00', '23|59|59', '09|30|00']
    result = []
    sts_list = []
    for i in tqdm(range(len(df1))):
        for j in time_patterns:
            if is_go_through(df1.iloc[i].TIMESTAMP, df1.iloc[i].TRIP_TIME, j):
                result.append(df1.iloc[i])
                sts_list.append(get_second_to_shapshot(df1.iloc[i].TIMESTAMP, j))
                break
    df1 = pd.DataFrame(result)
    df1['STS'] = sts_list

    df3 = df3.loc[(df3.MISSING_DATA == False) & (df3.CALL_TYPE == 'B') & (df3.ORIGIN_STAND > 0)]
    print("the shape of type B data", df3.shape)
    # patterns = ["2014|08|14|", "2014|09|30|", "2014|10|06|", "2014|11|1|", "2014|12|21|"]
    weekday_patterns = ["4", "2", "1", "5", "0"]
    time_patterns = ['14|00|00', '04|30|00', '13|45|00', '23|59|59', '09|30|00']
    result = []
    sts_list = []
    for i in tqdm(range(len(df3))):
        weekday = get_weekday(df3.iloc[i].TIMESTAMP)
        if weekday in weekday_patterns:
            index = weekday_patterns.index(weekday)
            result.append(df3.iloc[i])
            sts_list.append(get_second_to_shapshot(df3.iloc[i].TIMESTAMP, time_patterns[index]))
    df3 = pd.DataFrame(result)
    df3['STS'] = sts_list

    stand_map = {}
    for i in tqdm(range(len(df4))):
        stand_map[df4.iloc[i].ID] = [df4.iloc[i].Latitude, df4.iloc[i].Longitude]

    def get_gps(df):
        long_list = []
        lati_list = []
        for i in tqdm(range(len(df))):
            if df.iloc[i].CALL_TYPE == 'B' and df.iloc[i].ORIGIN_STAND > 0:
                lati_list.append(stand_map[df.iloc[i].ORIGIN_STAND][0])
                long_list.append(stand_map[df.iloc[i].ORIGIN_STAND][1])
            else:
                long_list.append(np.nan)
                lati_list.append(np.nan)
        df['LONG'] = long_list
        df['LATI'] = lati_list

    get_gps(df1)
    get_gps(df3)
    get_sh(df1)
    get_sh(df3)

    def build_train_data_b(df):
        new_df = df.loc[:, ['WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                            'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5',
                            'WEEKDAY_6', 'DATT', 'TIMESTAMP',
                            'ORIGIN_STAND', 'SP', 'SH', 'LONG', 'LATI', 'STS', 'TRIP_TIME']]
        new_df = new_df.dropna(axis=0, how='any')
        train_Y = np.array(new_df['TRIP_TIME'])
        train_X = np.array(new_df.loc[:, ['WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                                          'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5',
                                          'WEEKDAY_6', 'DATT', 'TIMESTAMP',
                                          'ORIGIN_STAND', 'SP', 'SH', 'LONG', 'LATI', 'STS']])
        print("The shape of training data", train_X.shape)
        return train_X, train_Y

    def build_test_data_b(df):
        new_df = df.loc[:, ['TRIP_ID', 'WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                            'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5',
                            'WEEKDAY_6', 'DATT', 'TIMESTAMP',
                            'ORIGIN_STAND', 'SP', 'SH', 'LONG', 'LATI', 'STS']]
        new_df = new_df.dropna(axis=0, how='any')
        indexes = np.array(new_df['TRIP_ID'])
        test_X = np.array(new_df.loc[:, ['WEEKDAY_0', 'WEEKDAY_1', 'WEEKDAY_2',
                                         'WEEKDAY_3', 'WEEKDAY_4', 'WEEKDAY_5',
                                         'WEEKDAY_6', 'DATT', 'TIMESTAMP',
                                         'ORIGIN_STAND', 'SP', 'SH', 'LONG', 'LATI', 'STS']])
        print("The shape of test data", test_X.shape)
        return test_X, indexes

    X, Y = build_train_data_b(df1)
    test_X = build_test_data_b(df3)
    model = XGBRegressor()
    model.fit(X, Y)
    predicts = model.predict(test_X[0])
    res = submit2(predicts, test_X[1], "submission_b.csv")
    return res


if __name__ == '__main__':
    print("Begin to load the files...")
    file1, file2, file3, file4 = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4])

    df11 = pd.DataFrame(pd.read_csv(file1, header=0))
    df22 = pd.DataFrame(pd.read_csv(file2, header=0))
    df33 = pd.DataFrame(pd.read_csv(file3, header=0))
    df44 = pd.DataFrame(pd.read_csv(file4, header=0))

    df11 = df11[df11.MISSING_DATA == False]
    print("training data size:", df11.shape)
    df33 = df33[df33.MISSING_DATA == False]
    print("test data size:", df33.shape)

    if not (os.path.exists("df1.csv") and os.path.exists("df3.csv")):
        train_feature, test_feature = feature_engineering(df11, df33)
    else:
        train_feature, test_feature = pd.DataFrame(pd.read_csv("df1.csv", header=0)), \
                                      pd.DataFrame(pd.read_csv("df3.csv", header=0))
    print("Initialize the model for type A data...")
    df_a = train_model_a(train_feature, test_feature)
    print("Initialize the model for type B data...")
    df_b = train_model_b(train_feature, test_feature, df44)
    print("Initialize the model for type C data...")
    df_c = train_model_c(train_feature, test_feature)
    result_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
    result_df.to_csv("submission.csv", index=False)






