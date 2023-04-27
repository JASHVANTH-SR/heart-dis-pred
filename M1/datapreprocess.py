import pandas as pd
import numpy as np
import os

data = pd.read_csv(r"./dataset/heart.csv")

data = data.replace('?', np.nan)

data

print('********before missing value*********')
print(data.isna().sum())

data['cp']=data['cp'].fillna(data['cp'].mode()[0])
data['trestbps']=data['trestbps'].fillna(int(data['trestbps'].mean()))
data['chol']=data['chol'].fillna(int(data['chol'].mean()))
data['fbs']=data['fbs'].fillna(data['fbs'].mode()[0])
data['restecg']=data['restecg'].fillna(data['restecg'].mode()[0])
data['thalach']=data['thalach'].fillna(int(data['thalach'].mean()))
data['exang']=data['exang'].fillna(data['exang'].mode()[0])
data['oldpeak']=data['oldpeak'].fillna(int(data['oldpeak'].mean()))
data['slope']=data['slope'].fillna(data['slope'].mode()[0])
data['ca']=data['ca'].fillna(data['ca'].mode()[0])

print('*****after clear missing value*****')
print(data.isna().sum())

data.to_csv(os.path.join('./preprocess_dataset/Preprocessed_Dataset.csv'), index=False)


