
"""
@author: ataka
"""
import pandas as pd
import numpy as np
import os 
from matplotlib import pyplot as plt 
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

tvdata = pd.read_csv("tv_shows.csv")

#Data infos and basic operations
tvdata.head(6)
tvdata.info()
tvdata.describe()
tvdata = tvdata.drop("Unnamed: 0", axis=1)
#Data type edits
for i in ["Title","Year","Netflix","Hulu","Prime Video","Disney+"]:
    tvdata[i] = tvdata[i].astype("category")    

tvdata = tvdata.drop("type",axis=1)
tvdata["Rotten Tomatoes"] = tvdata['Rotten Tomatoes'].str.replace(r"\D", '')
tvdata["Rotten Tomatoes"] = tvdata["Rotten Tomatoes"].astype("float64")
Total_Year = len((tvdata["Year"].cat.categories))
years = tvdata["Year"].cat.categories

tvdata["Age"] = tvdata["Age"].str.replace(r"\D", '')
tvdata["Age"] = tvdata["Age"].astype("category")
Total_Target_Age = len((tvdata["Age"].cat.categories))
Ages = tvdata["Age"].cat.categories

np.where(pd.isnull(tvdata))
print(tvdata[tvdata.isnull().any(axis=1)]["Age"].head())
#nan values are not non for Age column, eiditing them as NaN
tvdata["Age"] = tvdata["Age"].replace("", np.nan)

#Renaming Prime Video column
tvdata.rename(columns={"Prime Video": "Prime_Video"}, inplace=True)

#Dividing all categorical variables into indexes to see corrupted or blank values
Total_Target_Netflix = len((tvdata["Netflix"].cat.categories))
Netflix = tvdata["Netflix"].cat.categories

Total_Target_Hulu = len((tvdata["Hulu"].cat.categories))
Hulu = tvdata["Hulu"].cat.categories

Total_Target_Prime_Video = len((tvdata["Prime_Video"].cat.categories))
Prime_Video = tvdata["Prime_Video"].cat.categories

Total_Target_Disney = len((tvdata["Disney+"].cat.categories))
Prime_Video_Disney = tvdata["Disney+"].cat.categories

#Continious variables
tvdata.rename(columns={"Rotten Tomatoes":"Rotten_Tomatoes"}, inplace=True)
#tvdata.info()
tvdata["Rotten_Tomatoes"] = tvdata["Rotten_Tomatoes"].div(10)
#tvdata.describe()

# Detecting Null columns
null_columns=tvdata.columns[tvdata.isnull().any()]
tvdata[null_columns].isnull().sum()
mrdata = tvdata.dropna()


### Set figure size
plt.figure(figsize=(16,7))
cor = sns.heatmap(tvdata.corr(), annot = True)

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));

tvdata[["IMDb","Rotten_Tomatoes"]]
ContRatings = tvdata.iloc[:,3:5]
f=plt.figure(figsize=(12,6))
plt.scatter(ContRatings.IMDb, ContRatings.Rotten_Tomatoes)
plt.title("IMDb vs Rotten Tomatoes")
plt.xlabel("IMDb Ratings")
plt.ylabel("Rotten Tomatoes Ratings")



# Importing the dataset
X = mrdata.iloc[:, 1:3].values
y = mrdata.iloc[:, 3].values
print(X)
print(y)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
r2 = regressor.score(X, y)
regressor.summary()


# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#how is fitting score
ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title("MLR Model(Target Age & Year of Production)")
plt.xlabel("IMDb Ranks")
plt.ylabel("Ratio")
#correlations
X_dataframe = pd.DataFrame(X)


#SVR Model
svrdata = tvdata.dropna()
X2 = svrdata.iloc[:, 1:3].values
y2 = svrdata.iloc[:, 3].values
print(X2)
print(y2)
y2 = y2.reshape(len(y2),1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X2 = sc_X.fit_transform(X2)
y2 = sc_y.fit_transform(y2)
print(X2)
print(y2)
X2 = np.array(ct.fit_transform(X2))
#Splitings sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train2, y_train2)

y_pred2 = regressor.predict(X_test2)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred2.reshape(len(y_pred2),1), y_test2.reshape(len(y_test2),1)),1))

#how is fitting score
ax1 = sns.distplot(y_test2, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred2, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title("SVR Model(Target Age & Year of Production)")
plt.xlabel("IMDb Ranks")
plt.ylabel("Ratio")

# Visualising the SVR results
plt.scatter(y_test2,y_pred2, color = 'blue')
plt.title("SVR Scatter Plot(Target Age & Year of Production)")
plt.xlabel("Scaled Real Values")
plt.ylabel("Scaled Predicted Values")


#Lets continue with how does platform effect to IMDb ratings
platformdata = tvdata.dropna(subset=["IMDb"])
platformdata.isnull().any()
#I need to drop columns with NaN values 
platformdata = platformdata.drop(["Title","Year","Age","Rotten_Tomatoes"],axis=1)
platformdata = platformdata[["Netflix","Hulu","Prime_Video","Disney+","IMDb"]] #Reordring data 
#platformdata.info()
#fitting linear model to see correlations. 
platformdata[["Netflix","Hulu","Prime_Video","Disney+"]] = platformdata[["Netflix","Hulu","Prime_Video","Disney+"]].apply(pd.to_numeric)
a = platformdata[["Netflix","Hulu","Prime_Video","Disney+"]]
#correlation matrix for platforms
sns.heatmap(a)
plt.title("correlations of Platforms")
plt.show()

#Platform Counting Plot
s = a.idxmax(axis=1)
s = s.to_frame()  
s.columns =["Platforms"]
s["Platforms"]

s["Platforms"].value_counts().plot.bar(title = "Number of Series in each Platform",
                                       color=["firebrick",'green', 'blue', 'black'])
plt.xlabel("Platforms")
plt.ylabel("Total Series")
plt.yticks(rotation=20)
plt.xticks(rotation=20)


#Data Preprocessing
X3 = platformdata.iloc[:, :-1].values
y3 = platformdata.iloc[:, -1].values
print(X3)
print(y3)
y3 = y3.reshape(len(y3),1)


# Splitting the dataset into the Training set and Test set
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
regressor3 = LinearRegression()
regressor3.fit(X_train3, y_train3)
r3 = regressor3.score(X3, y3)

# Predicting the Test set results
y_pred3 = regressor3.predict(X_test3)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred3.reshape(len(y_pred3),1), y_test3.reshape(len(y_test3),1)),1))
#how is fitting score
ax1 = sns.distplot(y_test3, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred3, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title("Multiple Linear Regression(Platforms)")
plt.xlabel("IMDb Ranks")
plt.ylabel("Ratio")
# There is no effect when we only compare them as platforms for multiple lineare regression

#Addıng all columns to imnprove prediction 
FinalModel = svrdata.drop(columns=["Title","Rotten_Tomatoes"])
FinalModel = FinalModel[["Year","Age","Netflix","Hulu","Prime_Video","Disney+","IMDb"]]
Xf = FinalModel.iloc[:, :-1].values
yf = FinalModel.iloc[:, -1].values
print(Xf)
print(yf)
yf = yf.reshape(len(yf),1)

ctf = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
Xf = np.array(ctf.fit_transform(Xf))
X_trainf, X_testf, y_trainf, y_testf = train_test_split(Xf, yf, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressorf = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressorf.fit(X_trainf, y_trainf)
#Predicting
Y_predf = regressorf.predict(X_testf)
np.set_printoptions(precision=2)
print(np.concatenate((Y_predf.reshape(len(Y_predf),1), y_testf.reshape(len(y_testf),1)),1))
#Plot for Random Forest Classification
ax1 = sns.distplot(y_testf, hist=False, color="r", label="Actual Value")
sns.distplot(Y_predf, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title("Random Forest Classification(via ALL Indepent Variables)")
plt.xlabel("IMDb Ranks")
plt.ylabel("Ratio")





