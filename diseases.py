import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error,make_scorer, precision_score, f1_score

warnings.filterwarnings("ignore")
import math

df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

df = df.drop_duplicates(keep='first')

df.reset_index(drop=True, inplace=True)

x_encoder = OneHotEncoder(sparse_output=False,drop='first')
y_encoder  = LabelEncoder()

df['Fever'] = x_encoder.fit_transform(df[['Fever']])
df['Cough'] =x_encoder.fit_transform(df[['Cough']])
df['Fatigue'] = x_encoder.fit_transform(df[['Fatigue']])
df['Difficulty Breathing'] =x_encoder.fit_transform(df[['Difficulty Breathing']])
df['Gender'] = x_encoder.fit_transform(df[['Gender']])
df['Blood Pressure'] = x_encoder.fit_transform(df[['Blood Pressure']])
df['Cholesterol Level'] = x_encoder.fit_transform(df[['Cholesterol Level']])

df['Disease'] = y_encoder.fit_transform(df[['Disease']])
df['Outcome Variable'] = y_encoder.fit_transform(df[['Outcome Variable']])

df.drop(columns=['Cholesterol Level'], inplace=True)
df.drop(columns=['Blood Pressure'], inplace=True)
df.drop(columns=['Disease'], inplace=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100 , criterion = 'entropy' , random_state = 0)
classifier.fit(X_train,y_train)

pickle.dump(classifier,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))