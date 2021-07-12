import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv(r"D:\Proj\parkinsons.data")
df = df.drop(['MDVP:Jitter(%)' ,'MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ', 'Jitter:DDP','name','MDVP:Fo(Hz)'], axis=1)
features=np.array(df.drop('status',axis=1))
labels=np.array(df['status'])
x_train , x_test , y_train , y_test = train_test_split(features,labels,test_size=0.2,random_state=7)
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
pca.fit(x_train)
x_pca = pca.transform(x_train)
x_pcatest=pca.transform(x_test)
from xgboost import XGBClassifier
model1=XGBClassifier()
model1.fit(x_pca,y_train)

pickle.dump(model1,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
