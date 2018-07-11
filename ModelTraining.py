import pandas as pd
import lightgbm as lgb
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
def Lasso_model(train_X,train_Y):
    '''
    Fit samples with quadratic model.
    :param train_X: 
    :param train_Y: 
    :return: 
    '''
    model = linear_model.Lasso(alpha=0.1)
    model.fit(train_X, train_Y)
    return model

def quadratic_model(train_X,train_Y):
    '''
    Fit samples with quadratic model.
    :param train_X: 
    :param train_Y: 
    :return: 
    '''
    from sklearn.preprocessing import PolynomialFeatures
    quadratic = PolynomialFeatures(degree=2)
    model = linear_model.Lasso(alpha=0.1)
    train_X = quadratic.fit_transform(train_X)  # Transfer to quadratic atrbutes
    model.fit(train_X, train_Y)
    return model

def decision_tree(train_X,train_Y,depth=3):
    '''
    Fit samples with decision tree model.
    :param train_X: 
    :param train_Y: 
    :return: 
    '''
    model = tree.DecisionTreeRegressor(max_depth=depth)
    model.fit(train_X, train_Y)
    return model

def random_forest(train_X,train_Y,depth=3):
    '''
    Fit samples with decision tree model.
    :param train_X: 
    :param train_Y: 
    :return: 
    '''
    model = RandomForestRegressor(max_depth=depth )
    model.fit(train_X, train_Y)
    return model



if __name__ == '__main__':
    #Read data from preprocessed files
    train_data=pd.DataFrame(pd.read_csv('processed_data/application_train(chi2_selected).csv'))
    train_data.drop('SK_ID_CURR', axis=1)
    train_Y = train_data['TARGET']
    train_data = lgb.Dataset(train_data.drop('TARGET', axis=1), label=train_Y, free_raw_data=True)


    test_data=pd.DataFrame(pd.read_csv('processed_data/application_test(chi2_selected).csv'))
    test_data=test_data.drop('SK_ID_CURR',axis=1)

    test_result=pd.DataFrame(pd.read_csv('original_data/sample_submission.csv'))

    #Train model
    params = {'boosting_type': 'gbdt',
              'max_depth': 10,
              'objective': 'binary',
              'nthread': 5,
              'num_leaves': 64,
              'learning_rate': 0.05,
              'max_bin': 512,
              'subsample_for_bin': 200,
              'subsample': 1,
              'subsample_freq': 1,
              'colsample_bytree': 0.8,
              'reg_alpha': 5,
              'reg_lambda': 10,
              'min_split_gain': 0.5,
              'min_child_weight': 1,
              'min_child_samples': 5,
              'scale_pos_weight': 1,
              'num_class': 1,
              'metric': 'auc'
              }
    lgbm = lgb.train(params,
                     train_data
                     )

    # Predict on test set and write to submit
    test_pred= lgbm.predict(test_data)
    test_result['TARGET'] = test_pred
    test_result.to_csv('lighrgbm_result.csv', index=False)
