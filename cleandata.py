import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_absolute_error

df=pd.read_csv("winequality.csv")

print(df.isnull().sum())

x=df[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y=df[["quality","alcohol"]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(x_train)
print(y_test)

model=LinearRegression()

model.fit(x_train,y_train)

pre=model.predict(x_test)

result=r2_score(y_test,pre)

print("r2_score :",result)

print("mae :",mean_absolute_error(y_test,pre))