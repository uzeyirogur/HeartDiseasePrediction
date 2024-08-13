#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

#Data Upload
data = pd.read_csv("data/ah_kalbim.csv")
"""
data = pd.get_dummies(data,columns=["gogus_agrı_tipi","Elektrokardiyografik_Ölcümü","egim","talasemi"],drop_first=True)
"""

#The correlation between variables Dec
corr = data.corr()["amac"].sort_values()
#corr2 = data2.corr()["amac"].sort_values()

data["amac"].hist(figsize=(12,12))
plt.show()

data_counts = data.amac.value_counts()       #amac kolonu sayıları
plt.figure(figsize=(10, 6))
sns.countplot(x='amac', data=data, palette='Set2')
plt.xlabel('Amaç Değişkeni')
plt.ylabel('Sayı')
plt.title('Miktar')
plt.show()

#Yaşa göre kalp hastalığı sıklığı
pd.crosstab(data.yas,data.amac).plot(kind="bar",figsize=(20,6))
plt.title('Yaşa Göre Kalp Hastalığı Grafiği')
plt.xlabel('Yaş')
plt.ylabel('Sayı')
plt.show()

pd.crosstab(data.cinsiyet,data.amac).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Yaşa Göre Kalp Hastalığı Grafiği')
plt.xlabel('Cinsiyet (0 = Kadın, 1 = Erkek)')
plt.legend(["Kalp Hastası Değil", "Kalp Hastası"])
plt.ylabel('Sayı')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="gogus_agrı_tipi", y="amac", data=data, ci=None)
plt.xlabel("Göğüs Ağrı Tipi")
plt.ylabel("Amaç")
plt.title("Göğüs Ağrı Tipi ve Amaç İlişkisi")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x="gogus_agrı_tipi", y="amac", data=data)
plt.xlabel("Göğüs Ağrı Tipi")
plt.ylabel("Amaç")
plt.title("Göğüs Ağrı Tipi ve Amaç İlişkisi")
plt.show()

#Input Output
x = data.drop(["amac"],axis=1).values
y = data["amac"].values

#Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)

#Scale
from sklearn.preprocessing import MinMaxScaler,StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Creating a model
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(64,activation="relu",kernel_initializer="random_normal"))
model.add(Dropout(0.5))
model.add(Dense(32,activation="relu",kernel_initializer="random_normal"))
model.add(Dense(1,activation="sigmoid",kernel_initializer="random_normal"))
model.compile(optimizer="adam",loss="binary_crossentropy")
earlystopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)
model.fit(x_train,y_train,epochs=300,batch_size=64,validation_data=(x_test,y_test),verbose=1,callbacks=[earlystopping])

#Loss History
loss_history = model.history.history["loss"]       #loss değerlerinin geçmişi

#Predict
predict = model.predict(x_test)
threshold = 0.5
binary_predictions = [1 if p > threshold else 0 for p in predict]

#Gerçek değerlerle Tahmin değerleri karşılaştırması
binary_predictions = pd.Series(binary_predictions)
y_test = pd.Series(y_test)
birlesim = pd.concat([y_test, binary_predictions], axis=1)
birlesim.columns = ["Gerçek Değerlerimiz","Tahminlerimiz"]

#Accuracy Values
from sklearn.metrics import f1_score,classification_report,accuracy_score
f1 = f1_score(y_test, binary_predictions)
cm = classification_report(y_test, binary_predictions)
acc = accuracy_score(y_test, binary_predictions)*100


#MODEL 2
model2 = Sequential()
model2.add(Dense(48,activation="relu",kernel_initializer="random_uniform"))
model2.add(Dropout(0.5))
model2.add(Dense(24,activation="relu",kernel_initializer="random_uniform"))
model2.add(Dense(1,activation="sigmoid",kernel_initializer="random_uniform"))
model2.compile(optimizer="sgd",loss="binary_crossentropy")
earlystopping2 = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)
model2.fit(x_train,y_train,epochs=300,batch_size=32,validation_data=(x_test,y_test),verbose=1,callbacks=[earlystopping2])

loss_history2 = model2.history.history["loss"]       #loss değerlerinin geçmişi

predict2 = model2.predict(x_test)

threshold = 0.5
binary_predictions2 = [1 if p > threshold else 0 for p in predict2]

acc2 = accuracy_score(y_test, binary_predictions2)*100
f12 = f1_score(y_test, binary_predictions2)
cm2 = classification_report(y_test, binary_predictions2)








