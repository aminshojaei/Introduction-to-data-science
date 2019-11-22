# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:51:07 2019

@author: a335s717
"""
#================================================== Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics




#=================================================== Importing Datasets
spi_matches = pd.read_csv('https://raw.githubusercontent.com/aminshojaei/Introduction-to-data-science/master/Major%20League/dataset/spi_matches.csv', low_memory = False)

spi_international_score = pd.read_csv('https://raw.githubusercontent.com/aminshojaei/Introduction-to-data-science/master/Major%20League/dataset/spi_global_rankings_intl.csv', low_memory = False)

spi_off_def = pd.read_csv('https://raw.githubusercontent.com/aminshojaei/Introduction-to-data-science/master/Major%20League/dataset/spi_global_rankings.csv', low_memory = False)



#=================================================== Cleaning Datasets
newdataset= spi_matches.loc[:,'team1':'score2']     # It sellect any rows and column 'team1' to 'score2'
first_spi_matches = newdataset.head()

newdataset=newdataset.dropna(how='any')      # for removing NaN data rows

statics = newdataset.describe()     # some quick summary

#==================================================== Merging Data
merged_dataset_1 = pd.merge(spi_off_def[['name','off']] ,newdataset, left_on='name' , right_on='team1' , how='left' )
merged_dataset=pd.merge(merged_dataset_1 ,spi_off_def[['name','off']] , left_on='team2' , right_on='name' , how='right' )
merged_dataset.head()

merged_dataset.rename(columns={'off_x':'offensive_team1'}, inplace=True)
merged_dataset.rename(columns={'off_y':'offensive_team2'}, inplace=True)

merged_dataset=merged_dataset[['team1','team2','offensive_team1','offensive_team2','spi1','spi2','prob1','prob2', 'probtie', 'proj_score1', 'proj_score2', 'importance1','importance2', 'score1', 'score2']]




#=================================================Plot some features
sb.heatmap(merged_dataset.corr())    # Heatmap

sb.pairplot(merged_dataset, vars=["offensive_team1","offensive_team2","spi1","spi2","proj_score1","proj_score2","prob1","prob2","score1","score2"])



#================================================regression model
X=merged_dataset[['spi1','spi2','prob1','prob2','importance1','importance2','offensive_team1','offensive_team2',"proj_score1","proj_score2"]].values
Y=merged_dataset['score1'].values
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
regressor= LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)


df=pd.DataFrame({'Actual': y_test , 'Prediction': y_pred})

print(regressor.coef_)
er=[]
for i in range(len(y_test)):
    x=(y_test[i]-y_pred[i])**2
    er.append(x)

print('variance is:  %.2f'% np.var(er))
print('regression intercept is: %.2f'%regressor.intercept_)
print('R2 score : %.2f'% r2_score(y_test,y_pred))
print('Mean absolute error: %.2f'% metrics.mean_absolute_error(y_test,y_pred))
print('Mean squared error: %.2f'% metrics.mean_squared_error(y_test,y_pred))
print('root mean squared error: %.2f'% np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

compare= pd.DataFrame ({'actual value':y_test , 'prediction': y_pred})

df = compare.head(100)
error = df['actual value'] - df['prediction']
plt.plot(error)
plt.ylabel('error')
plt.xlabel('First 100 data')
df.plot(kind='bar',figsize=(10,10))



#==================================================Random forest Model

merged_dataset['for_train']=np.random.uniform(0,1,len(merged_dataset)) <= 0.8
merged_dataset.head()


train , test = merged_dataset[merged_dataset['for_train']==True] , merged_dataset[merged_dataset['for_train']==False]

features = merged_dataset.columns[2:13]

y1_train=train['score1']
y2_train=train['score2']
y1_test=test['score1']
y2_test=test['score2']

clf=RandomForestClassifier( n_jobs=2 , random_state=0)
clf.fit(train[features],y1_train)
clf.fit(train[features],y2_train)

clf.predict(test[features])

# viewing the predicted probabilities of the first 10 observations
clf.predict_proba(test[features])[0:10]



