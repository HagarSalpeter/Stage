# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:06:34 2022

@author: hagar
"""

# # 3. Train Custom Model Using Scikit Learn

# ## 3.1 Read in Collected Data and Process

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle # file to save the trained model


# In[2]:


df = pd.read_csv(r"C:\Users\hagar\OneDrive - mail.tau.ac.il\Desktop\Stage\LPC_2022\Hand decoder\Position\data\coords_position.csv")


# Separate the features from the target
X = df.drop('class', axis=1) # features
y = df['class'] # target value



# Split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



# ## 3.2 Train Machine Learning Classification Model

# Use different classifiers and see what works best
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    #'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}





fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model





fit_models





fit_models['rc'].predict(X_test)


# ## 3.3 Evaluate and Serialize Model 

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))


# In[13]:

#run with specific model
fit_models['rc'].predict(X_test)






# Put the trained model in a pkl file
with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rc'], f)