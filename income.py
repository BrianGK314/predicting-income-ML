def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv("income.csv",header=0,delimiter = ', ')
income_data['sex-int'] = income_data['sex'].apply(lambda x:1 if x == 'Female' else 0)
income_data['country-int'] = income_data['native-country'].apply(lambda x:0 if x == 'United-States' else 1)
income_data['race-int'] = income_data['race'].apply(lambda x: 0 if x == 'white' else 0)

#print(income_data.iloc[5])
#print(income_data["race"].value_counts())

labels = income_data[['income']]
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week",'sex-int','country-int','race-int']]


train_data,test_data,train_labels,test_labels = train_test_split(data,labels,random_state=1)

forest = RandomForestClassifier(random_state = 1)
forest.fit(train_data,train_labels)
score = forest.score(test_data,test_labels)
print(score)
print(forest.feature_importances_)

m = tree.DecisionTreeClassifier(random_state =222)
m.fit(train_data,train_labels)
score = m.score(test_data,test_labels)
print(score)
