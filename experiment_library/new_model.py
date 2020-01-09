# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.externals import joblib

def encoding(name):
    '''

    :param name: The full dir of the csv file to be read
    :return: The data after one-hot encode
    '''



    data=pd.read_csv(name)
    #类别向量处理
    for i in ['renovationCondition','buildingStructure','elevator','fiveYearsProperty',
              'buildingType','floornumer']:
        category = pd.Categorical(data[i])
        data[i]=category.codes

    return data


def preprocessing(features_train,features_test):
    '''

    :param features_train: The features of train dataset
    :param features_test: The features of test dataset
    :return: data after standard scale
    '''

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_train)
    X_test = scaler.transform(features_test)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    return X_train,X_test


def get_model(model_name='Random Forest',n_estimators=80):
    '''

    :param model_name: The model name of the machine learning model,only Random Forest supported now
    :param n_estimators: The estimator number of RF
    :return:
    '''
    if model_name == 'Random Forest':
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            print('import error')
            return

        model = RandomForestRegressor(n_estimators=n_estimators,random_state=0)
    else:
        raise NotImplementedError
    return model

def train_model(model,X_train,y_train,kfold_number = 5):
    '''

    :param model: The sklearn model to be trained
    :param X_train: The features of the train dataset
    :param y_train: The target of the train dataset
    :param kfold_number: The number of the k-fold
    :return:
    '''
    kf = KFold(kfold_number)
    for train_index, test_index in kf.split(X_train):
        start = test_index[0]
        end = test_index[-1]

        X_train_s, X_test_s = pd.concat([X_train[:start],X_train[end+1:]]), X_train[start:end+1]
        y_train_s, y_test_s = pd.concat([y_train[:start],y_train[end+1:]]), y_train[start:end+1]
        #print(X_train_s.shape, X_test_s.shape)
        model.fit(X_train_s,y_train_s)
    trained_model = model
    return trained_model

def main():
    # one hot encoding
    data_train = encoding('./cleandata_Train.csv')
    data_test = encoding('./cleandata_Test.csv')

    # some features that are not necessary are droped
    drop_features = [data_train.columns[0],'price','company','livingragion','transroutes','busstation','officebuilding',
          'dormroom','entertainment','hospital','government','teaching','culturemedia',
          'tourist','food','shopping','exercise','hotel','financial']
    X_train,y_train = data_train.drop(drop_features,axis=1) ,data_train.price
    X_test, y_test = data_test.drop(drop_features, axis=1), data_test.price
    # using StandardScaler to preprocessing the feature
    X_train,X_test = preprocessing(X_train,X_test)
    print(X_train.shape,X_test.shape)



    print('开始训练【model】随机森林')
    trained_model = train_model(get_model(model_name='Radom Forest',n_estimators=80),X_train,y_train)
    print('-------------训练完成------------')
    y_pre= trained_model.predict(X_test)
    print(' :',mean_squared_error(y_pre,y_test))

    print('-------------保存模型------------')
    joblib.dump(trained_model,'rf.check_point')
