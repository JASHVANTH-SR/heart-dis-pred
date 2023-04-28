import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
from imblearn.pipeline import make_pipeline

st.set_page_config(page_title='Heart Disease Prediction ❤️🫀',layout="wide", initial_sidebar_state="auto", menu_items=None)


# Get rid of pesky warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.warn = warn

# Define column names
column_names = [
            "Age:",  #1
            "Sex:",  #2
            "CP:",  #3
            "Trest-BP:",  #4
            "Chol:",  #5
            "FBS:",   #6
            "Rest-ECG:",  #7
            "Thalach:",  #8
            "Exang:", #9
            "Oldpeak:",  #10
            "Slope:",  #11
            "CA:", #12
            "Thal:", #13
            "target"  #14
        ]

# Load the dataset
location = 'M3/dataset/Preprocessed_Dataset.csv'
dataset = pd.read_csv(location)
dataset = dataset.sample(frac=1).reset_index(drop=True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.3)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Use Pipeline
clf = MLPClassifier(solver='lbfgs', learning_rate='constant', activation='tanh')
kernel = KernelPCA()

pipeline = make_pipeline(kernel, clf)
pipeline.fit(X_train, Y_Train)

# User-input
st.title("Heart Disease Possibility Prediction")

st.markdown("### Please enter the following details to predict the possibility of a heart attack.")
if st.button("Show Sample Values"):
    with open('M3/Test_Data/yes Heart disease data.txt') as f:
        lines = f.readlines()
    for line in lines:
        st.write(line.strip())
    st.write('\n')        

v = []

for i in column_names[:-1]:
    v.append(st.text_input(i, key=i))

if st.button("Predict"):
    answer = np.array(v)
    answer = answer.reshape(1, -1)
    answer = sc_X.transform(answer)

    prediction = pipeline.predict(answer)

    if prediction == 1:
        st.markdown("## Sorry, Please consider consulting a doctor.")
        st.image('M3/heartdis.jpg')
    else:
        st.markdown("## Congratulations! Your heart is healthy.")
        st.image('M3/heartheal.jpg')
if st.button('show app credits'):
  st.markdown('''##### This is the **Social Awareness App** created in Streamlit using the **PySpark and Scikit-Learn** library.
##### **Credit:** App built in `Python` + `Streamlit` by [HARUL GANESH S B ](https://www.linkedin.com/in/harul-ganesh/)\t\t\t[BALAJI S ](https://www.linkedin.com/in/balaji-suresh-kumar/)
---
''')
