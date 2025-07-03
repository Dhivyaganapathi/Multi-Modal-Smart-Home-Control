import mediapipe as mp 
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import RidgeClassifier

df = pd.read_csv('dataset.csv')
df.head()
X = df.drop('class', axis=1) 
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


pipelines = {
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
}
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

fit_models['rc'].predict(X_test)

from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle

for algo, model in fit_models.items():
    yhat = model.predict(X_test)

fit_models['rc'].predict(X_test)

with open('model.pkl', 'wb') as f:
    pickle.dump(fit_models['rc'], f)
    print('model saved')
        
   
