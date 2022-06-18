#assignment2

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#reading dataset & making copy
train= pd.read_csv('F:/ml/ml/train.csv')
test= pd.read_csv('F:/ml/ml/test.csv')

#first 5 columns
train.head()
#last 5 columns
train.tail()
#columns' type
train.dtypes
#when it runs if numbers of (non-null) > num of entries ,this means that there is null 
train.info()



corr=train.corr()
corr
corr["SalePrice"]


###################model###########
lr  = LinearRegression()
#2-features
X = pd.DataFrame(np.c_[train['GrLivArea'], train['OverallQual']], columns = ['GrLivArea','OverallQual'])
y = train[["SalePrice"]]



x_train, x_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3,random_state=5)


model = lr.fit(x_train, Y_train)
y_predict=model.predict(x_test)



final=test.loc[:,['GrLivArea','OverallQual']]
y_predict_test= model.predict(final)
Finaal  = pd.DataFrame()
Finaal['Id']=test['Id']
Finaal["SalePrice"]=y_predict_test
Finaal.info()
Finaal.to_csv('submission.csv', index=False) 


Finaal.head()
Finaal.tail()


