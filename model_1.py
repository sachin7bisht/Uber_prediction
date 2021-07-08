import pandas as pd
import numpy as np
import pickle
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('taxi.csv')
# print(data.head())
data_x=data.iloc[:,0:-1].values
data_y=data.iloc[:,-1].values

train_x,test_x,train_y,test_y=train_test_split(data_x,data_y,test_size=0.3,random_state=42)

reg=LinearRegression()
reg.fit(train_x,train_y)

print('train score',reg.score(train_x,train_y))
print('test score',reg.score(test_x,test_y))

pickle.dump(reg,open('taxi.pkl','wb'))

model=pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80,1070070,6000,85]]))