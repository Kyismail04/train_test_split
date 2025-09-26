import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df=pd.read_csv('audi.csv')

df=df[["Year","Type","Mileage(miles)","Engine","PS","Transmission","Fuel","Number_of_Owners","Price(£)"]]

df.columns=["yil","kasa","mil","motor","ps","vites","yakit","sahip","fiyat"]

df['motor']=df['motor'].str.replace("L","")
df['motor']=pd.to_numeric(df['motor'])

df=pd.get_dummies(df,columns=['kasa','vites','yakit'],drop_first=True)

y=df["fiyat"]
x=df.drop('fiyat',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=22)

lr=LinearRegression()
model=lr.fit(x_train,y_train)

print("tahmin: ",model.predict([[2016,30000,1.0,90,5,0,1]]))

print("skor: ", model.score(x_test,y_test))