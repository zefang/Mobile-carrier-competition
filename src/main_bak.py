'''

'''

import numpy as np
import pandas as pd
# import lightgbm as lgb
# import warnings
# warnings.filterwarnings('ignore')
import LightGBM_KFold as lgb
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path",                  default='../data/',            help="path of trainset and testset")
parser.add_argument("--model",                  default='s3',            help="saved model name")
# parser.add_argument("netType",      choices=["CNN","STN","IC-STN"],     help="type of network")
# parser.add_argument("--transScale", type=float, default=0.25,           help="initial translation scale")
opt = parser.parse_args()

path = opt.path
modelname = opt.model

# train = pd.read_csv(path + 'datas_8classes.csv')
train = pd.read_csv(path + 'datas.csv')
# test = pd.read_csv(path + 'test_datas_8classes.csv')
test = pd.read_csv(path + 'test_datas.csv')

print(set(train.columns)-set(test.columns))
print('train data shape',train.shape)
print('train data of user_id shape',len(set(train['user_id'])))
print('train data of current_service shape',(set(train['current_service'])))
print('train data shape',test.shape)
print('train data of user_id shape',len(set(test['user_id'])))

# sm mean sub many_over_bill==1;n4 means only net_service==4;
if modelname == "n4":
  train = train[train['net_service']==4]
elif modelname == "n3n4":
  index=(train['net_service']==4) | (train['net_service']==3)
  train = train[index]
elif modelname == "n2n3n4":
  index=(train['net_service']==4) | (train['net_service']==3) | (train['net_service']==2)
  train = train[index]
elif modelname == "sm":
  index=(train['many_over_bill']==0)
  train = train[index]
elif modelname == "m1":
  index=(train['many_over_bill']==1)
  train = train[index]

classes = sorted(list(set(train['current_service'])))
label2current_service = dict(zip(range(0,len(classes)), classes))
current_service2label = dict(zip(classes,range(0,len(classes))))

train['current_service'] = train['current_service'].map(current_service2label)

y = train.pop('current_service')
train_id = train.pop('user_id')

X = train
train_col_tmp = ['former_complaint_fee','former_complaint_num','complaint_level']
# train_col_tmp = ['complaint_level', '2_total_fee','3_total_fee','4_total_fee','net_service', 'former_complaint_num', 'former_complaint_fee',\
 # 'pay_times', 'local_caller_time', 'is_mix_service', 'is_promise_low_consume']

for itm in  train_col_tmp:
    train.pop(itm)
    print(itm)

train_col = train.columns
X_test = test[train_col]
test_id = test['user_id']

X.replace("\\N",-1)
X_test.replace("\\N",-1)

X,y,X_test = X.values,y.values,X_test.values

cv_pred = lgb.lightgbm_kfold(X, y, X_test, classes, modelname)

submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))

df_test = pd.DataFrame()
df_test['id'] = list(test_id.unique())
df_test['predict'] = submit
df_test['predict'] = df_test['predict'].map(label2current_service)

df_test.to_csv('../result/baseline2_%s.csv'%(modelname),index=False)

if __name__ == '__main__':
  print('main')