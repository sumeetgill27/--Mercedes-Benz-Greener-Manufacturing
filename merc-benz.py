import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
merc_test = pd.read_csv("test.csv")
merc_train = pd.read_csv("train.csv")

#Visualising Data
plt.scatter(range(len(merc_train)), np.sort(merc_train.y.values), alpha=0.3)
    
y_train = merc_train.iloc[:,1].values
x_train = merc_train.iloc[:,2:378].values

x_test = merc_test.iloc[:,1:377].values

#Encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


i = 0
while i <= 9:
    le_train = LabelEncoder()
    le_train.fit_transform(x_train[:,i])
    x_train[:,i]=le_train.fit_transform(x_train[:,i])
    i +=1
    
j = 0
while j <= 8:
    le_test = LabelEncoder()
    le_test.fit_transform(x_test[:,j])
    x_test[:,j]=le_test.fit_transform(x_test[:,j])
    j +=1

#Variance
x1_train = pd.DataFrame(x_train)
x1_test = pd.DataFrame(x_test)
v_test = x1_test.var()
v_train = x1_train.var()
y1_train = pd.DataFrame(y_train)
vy_train = y1_train.var()

# Adding Variance row
x1_test = x1_test.append(v_test, ignore_index = True)
x1_train= x1_train.append(v_train, ignore_index = True)
    
#Dropping columns with zero variance
x2_train=x1_train.drop(x1_train.columns[x1_train.loc[4209,:] == 0],axis=1)
x2_test=x1_test.drop(x1_test.columns[x1_test.loc[4209,:] == 0],axis=1)

#Check for null and unique values
x1_train.isnull().any().any()
x1_test.isnull().any().any()
y1_train.isnull().any().any()
x1_train.nunique(dropna=True)
x1_test.nunique(dropna=True)
y1_train.nunique(dropna=True)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1_test = sc.fit_transform(x1_test)
x1_train = sc.fit_transform(x1_train)
y1_train = sc.fit_transform(y1_train)

#Apply Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
x_test_pca = pca.fit_transform(x1_test)
x_train_pca = pca.transform(x1_train)
explained_variance = pca.explained_variance_ratio_


#Splitting data in cross-validation set and training set

from sklearn.model_selection import train_test_split
xg_train, xg_cv, y_xg_train, y_xg_cv = train_test_split(x_train, y_train, test_size=0.3,random_state = 0)

#Using XGBOOST Model
import xgboost as xgb
from xgboost import XGBRegressor
Reg = XGBRegressor()
Reg.fit(xg_train,y_xg_train)

#Predicting for test dataset
y_cvpred = Reg.predict(xg_cv)
y_test = Reg.predict(x_test)

plt.scatter(range(len(xg_cv)), np.sort(y_cvpred), alpha=0.3)

#Accuracy
from sklearn.metrics import r2_score
print(r2_score(y_xg_cv, y_cvpred))

