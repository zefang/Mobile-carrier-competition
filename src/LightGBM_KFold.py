#!/usr/bin/python
# -*- coding: UTF-8 -*-

# import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.externals import joblib
# import matplotlib.pyplot as plt

# def draw_importance(features, importances):
    # indices = np.argsort(importances)
    # print(indices)
    # print(features)
    # plt.title('Feature Importances')
    # plt.barh(range(len(indices)), np.array(importances)[indices], color='b', align='center')
    # plt.yticks(range(len(indices)), np.array(features)[indices])
    # plt.xlabel('Relative Importance')
    # plt.show()

def train_and_save(params, X, y, X_test, classes, n_splits, seed, modelname):
  xx_score = []
  cv_pred = []

  def f1_score_vali(preds, data_vali):
      labels = data_vali.get_label()
      preds = np.argmax(preds.reshape(len(classes), -1), axis=0)
      score_vali = f1_score(y_true=labels, y_pred=preds, average=None)
      # print(score_vali)
      score_vali = np.mean(score_vali)
      score_vali = score_vali * score_vali
      return 'f1_score', score_vali, True

  skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
  for index,(train_index,test_index) in enumerate(skf.split(X,y)):
      print(index)

      X_train,X_valid,y_train,y_valid = X[train_index],X[test_index],y[train_index],y[test_index]

      train_data = lgb.Dataset(X_train, label=y_train)
      validation_data = lgb.Dataset(X_valid, label=y_valid)

      clf=lgb.train(params,train_data,num_boost_round=1800,valid_sets=[validation_data],early_stopping_rounds=100,feval=f1_score_vali,verbose_eval=False)

      xx_pred = clf.predict(X_valid,num_iteration=clf.best_iteration)
      
      xx_pred = [np.argmax(x) for x in xx_pred]
      online_score = f1_score(y_valid,xx_pred,average=None)
      print('best_iteration is: %d, online score is: '%(clf.best_iteration), online_score)
      online_score= np.mean(online_score)
      online_score= online_score*online_score
      xx_score.append(online_score)

      y_test = clf.predict(X_test,num_iteration=clf.best_iteration)

      y_test = [np.argmax(x) for x in y_test]

      if index == 0:
          cv_pred = np.array(y_test).reshape(-1, 1)
      else:
          cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
      ##feature importance#############
      #importance = clf.feature_importances_
      # importance = clf.feature_importance()
      # print(importance)
      # indices = np.argsort(importance)[::-1]
      # print("----the importance of features and its importance_score------")
      # j = 1
      # features_names = []
      # im_list = []
      # features_num = len(train_col)
      # for i in indices[0:features_num]:
          # f_name = train_col.values[i]
          # print(j, f_name, importance[i])
          # features_names.append(train_col.values[i])
          # im_list.append(importance[i])
          # j += 1

      #draw_importance(features_names, im_list)
      #break
      ######end########################

  # lgb.plot_importance and save https://blog.csdn.net/m0_37477175/article/details/80567010
  print(xx_score,np.mean(xx_score))    
  # save model
  joblib.dump(clf,'../models/lgb_%s.pkl'%(modelname))
  return cv_pred
  
def lightgbm_kfold(X, y, X_test, classes, modelname):
  n_splits = 5
  seed = 42
  params={
      'subsample': 0.975036929296666,
      'reg_lambda': 0.4711661128684218,
      'reg_alpha': 0.5097390210129251, 
      'min_child_samples': 25, 
      'colsample_bytree': 0.6520278612930721, 
      'n_jobs': 4,
      "learning_rate":0.1,
      "max_depth":-1,
      # "min_child_samples":20,
      # "min_child_weight":0.001, 
      # "min_split_gain":0.0, 
      # "n_estimators":100,
      "objective":"multiclass",
      "num_class":len(classes),
      "is_unbalance":True,
      "verbosity":-1,
      "device":'gpu',
      'subsample_for_bin': 40000,
      "num_leaves":103,
      # "subsample_freq":1
      # "gpu_platform_id":-1,
      # "gpu_device_id":-1
      # 'gpu_use_dp':False
  } 
  cv_pred = train_and_save(params, X, y, X_test, classes, n_splits, seed, modelname)
  return cv_pred

if __name__ == '__main__':
  print('LightGBM_KFold')