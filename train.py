# =============================================================================
# Task 3 - Integrated Diagnosis
# Miccai hackathon 2022

# Tomé and Sajid
# =============================================================================


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modality', choices=['temp','blood','radiology','mean_std_all','mean_all'], default='temp')
parser.add_argument('--multiclass', choices=['True','False'], default='True')
args = parser.parse_args()

#imports
import pandas as pd
import numpy as np

# import shap
from xgboost import XGBRegressor
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error



# load data

data_train = pd.read_csv('data/train_'+str(args.modality)+'.csv')
data_val = pd.read_csv('data/val_'+str(args.modality)+'.csv')
data_test = pd.read_csv('data/test_'+str(args.modality)+'.csv')

# frames = [data_train, data_val]
  
# data_train = pd.concat(frames)

# split data into X and y
X_train = data_train.drop(['Patient','risk_1', 'risk_2', 'risk_3'], axis=1) #subset de treino do modelo
Y_train = data_train.loc[:,['risk_1', 'risk_2', 'risk_3']] # subset de validação do modelo

X_val = data_val.drop(['Patient','risk_1', 'risk_2', 'risk_3'], axis=1) #subset de treino do modelo
Y_val = data_val.loc[:,['risk_1', 'risk_2', 'risk_3']] # subset de validação do modelo

X_test = data_test.drop(['Patient','risk_1', 'risk_2', 'risk_3'], axis=1) #subset de treino do modelo
Y_test = data_test.loc[:,['risk_1', 'risk_2', 'risk_3']] # subset de validação do modelo

from sklearn.preprocessing import LabelEncoder
#
# Instantiate LabelEncoder
#
le = LabelEncoder()

cols = ['risk_1', 'risk_2', 'risk_3']

"""
Data encoding 

100 -> 1
010 -> 2
001 -> 3
110 -> 4
101 -> 5
011 -> 6
111 -> 7

"""

# Encode labels of multiple columns at once
#
if args.multiclass == 'True':
    
    Y_train[cols] = Y_train[cols].apply(LabelEncoder().fit_transform)
    
    labels=[[Y_train.iloc[i,0],Y_train.iloc[i,1], Y_train.iloc[i,2]] for i in range(len(Y_train))]
    
    Y_trainn=[]
    for l in labels:
        if l == [1,0,0]:
            Y_trainn.append(1)
        elif l== [0,1,0]:
            Y_trainn.append(2)
        elif l== [0,0,1]:
            Y_trainn.append(3)
        elif l== [1,1,0]:
            Y_trainn.append(4)
        elif l== [1,0,1]:
            Y_trainn.append(5)
        elif l== [0,1,1]:
            Y_trainn.append(6)
        elif l== [1,1,1]:
            Y_trainn.append(7)
        else:
            Y_trainn.append(0)
            
    Y_test[cols] = Y_test[cols].apply(LabelEncoder().fit_transform)
    
    labels=[[Y_test.iloc[i,0],Y_test.iloc[i,1], Y_test.iloc[i,2]] for i in range(len(Y_test))]
    
    Y_testt=[]
    for l in labels:
        if l == [1,0,0]:
            Y_testt.append(1)
        elif l== [0,1,0]:
            Y_testt.append(2)
        elif l== [0,0,1]:
            Y_testt.append(3)
        elif l== [1,1,0]:
            Y_testt.append(4)
        elif l== [1,0,1]:
            Y_testt.append(5)
        elif l== [0,1,1]:
            Y_testt.append(6)
        elif l== [1,1,1]:
            Y_testt.append(7)
        else:
            Y_testt.append(0)
            
else :
    
    Y_train[cols] = Y_train[cols].apply(LabelEncoder().fit_transform)
    
    labels=[[Y_train.iloc[i,0],Y_train.iloc[i,1], Y_train.iloc[i,2]] for i in range(len(Y_train))]
    
    Y_trainn=[]
    for l in labels:
        if l == [1,0,0]:
            Y_trainn.append(1)
        elif l== [0,1,0]:
            Y_trainn.append(1)
        elif l== [0,0,1]:
            Y_trainn.append(1)
        elif l== [1,1,0]:
            Y_trainn.append(1)
        elif l== [1,0,1]:
            Y_trainn.append(1)
        elif l== [0,1,1]:
            Y_trainn.append(1)
        elif l== [1,1,1]:
            Y_trainn.append(1)
        else:
            Y_trainn.append(0)
            
    Y_test[cols] = Y_test[cols].apply(LabelEncoder().fit_transform)
    
    labels=[[Y_test.iloc[i,0],Y_test.iloc[i,1], Y_test.iloc[i,2]] for i in range(len(Y_test))]
    
    Y_testt=[]
    for l in labels:
        if l == [1,0,0]:
            Y_testt.append(1)
        elif l== [0,1,0]:
            Y_testt.append(1)
        elif l== [0,0,1]:
            Y_testt.append(1)
        elif l== [1,1,0]:
            Y_testt.append(1)
        elif l== [1,0,1]:
            Y_testt.append(1)
        elif l== [0,1,1]:
            Y_testt.append(1)
        elif l== [1,1,1]:
            Y_testt.append(1)
        else:
            Y_testt.append(0)
# Model
model = XGBRegressor()
model = XGBRegressor(n_estimators=2000,random_state=10, max_depth=150, eta=0.1, subsample=0.8, colsample_bytree=0.8)

# treinar o modelo com o subset de treino
model.fit(X_train,Y_trainn)

print(model)

# correr o modelo sem treinar para obter previsões para o subset de validação
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_testt, pred))
MAE = mean_absolute_error(Y_testt, pred)
print("RMSE : % f" %(rmse))
print("MAE : % f" %(MAE))

from sklearn.metrics import classification_report

print(classification_report(Y_testt, np.rint(pred)))


import time

t = time.time()
import shap
shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
# do stuff
elapsed = time.time() - t
