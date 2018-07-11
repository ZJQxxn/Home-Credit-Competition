'''
Description: 
    Regression model training for Home Credit contest.(https://www.kaggle.com/c/home-credit-default-risk)
Author:
    Jiaqi   Zhang
'''
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import random

def feature_type_split(data):
    '''
    Split all categories into 3 types.
    :param data: 
    :param special_list: 
    :return: 
    '''
    cat_list = []
    dis_num_list = []
    num_list = []
    for i in data.columns.tolist():
        if data[i].dtype == 'object':
            cat_list.append(i)
        elif data[i].nunique() < 10:
            dis_num_list.append(i)
        else:
            num_list.append(i)
    return cat_list, dis_num_list, num_list

if __name__ == '__main__':
    # Read bureau
    bureau = pd.read_csv('original_data/bureau.csv')
    bureau=bureau.head(200)
    bureau=pd.get_dummies(bureau)
    bureau = pd.groupby('SK_ID_CURR')
    print('')
    '''
    #Read training data and split target and attrbutes
    train=pd.read_csv('original_data/application_train.csv')
    train_Y=train['TARGET']
    train_ID=train['SK_ID_CURR']
    train_X=train.drop('TARGET',axis=1).drop('SK_ID_CURR',axis=1)
    del train

    # Read testing data and split ID and attrbutes
    test=pd.read_csv('original_data/application_test.csv')
    test_ID=test['SK_ID_CURR']
    test_X=test.drop('SK_ID_CURR',axis=1)
    del test

   
    #Filling missing values
    cat_list, dis_num_list, num_list = feature_type_split(train_X)
    for cat in cat_list:
        while True:
            index=random.randint(0,len(train_X[cat]))-1
            temp=train_X[cat][index] if index>=0 else train_X[cat][0]
            if not pd.isna(temp):
                break
        train_X[cat]=train_X[cat].fillna(temp)
        test_X[cat] = test_X[cat].fillna(temp)
    for dis in dis_num_list:
        while True:
            index = random.randint(0, len(train_X[dis])) - 1
            temp = train_X[dis][index] if index >= 0 else train_X[dis][0]
            if not pd.isna(temp):
                break
        train_X[dis] = train_X[dis].fillna(temp)
        test_X[dis] = test_X[dis].fillna(temp)
    for num in num_list:
        train_X[num] = train_X[num].fillna(train_X[num].mean())
        test_X[num] = test_X[num].fillna(test_X[num].mean())

    #Convert categorical value to one-hot
    train_X['is_train']=1
    test_X['is_train']=0
    total_x=train_X.append(test_X)
    total_x=pd.get_dummies(total_x)
    train_X=total_x[total_x['is_train']==1].drop('is_train',axis=1)
    test_X=total_x[total_x['is_train']==0].drop('is_train',axis=1)
    del total_x

    # Feature selection
    chi_selector = SelectKBest(chi2, k=100)
    chi_selector.fit(MinMaxScaler().fit_transform(train_X), train_Y)
    chi_support = chi_selector.get_support()
    chi_feature = train_X.loc[:, chi_support].columns.tolist()
    print(chi_feature)

    train_X=train_X[chi_feature]
    train_X['SK_ID_CURR']=train_ID
    train_X['TARGET']=train_Y

    test_X = test_X[chi_feature]
    test_X['SK_ID_CURR'] = test_ID

    train_X.to_csv('processed_data/application_train(chi2_selected).csv',index=False)
    test_X.to_csv('processed_data/application_test(chi2_selected).csv', index=False)
    '''
