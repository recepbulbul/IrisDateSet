import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv("Iris.csv")

x = ds.iloc[:,1:-1]
y = ds.iloc[:,-1]

purple ='#2d023d'
ten = '#ffd4aa'
pink = '#ffcbdb'

from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test =train_test_split(x,y, test_size=0.2, random_state=121)

print(ds.head())
print(ds.info())
print(ds.describe())

print(ds["Species"].value_counts())

print(ds.isnull().sum())

correlation_matrix  = x.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", linewidths=.10)
plt.title('Özellikler Arasındaki Korelasyon Heatmap')
plt.show()
plt.figure(figsize = (15, 15))
for i in enumerate(list(ds.columns[1:-1])):
    plt.subplot(3, 3,i[0]+1)

    # Histlot plotting the fetures in the dataset
    sns.histplot(
        data = x , 
        x = x[i[1]],
        hue = y,
        palette= [ten, purple,pink], 
        kde = True, 
        multiple='stack', 
        alpha=1
    )


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)

from sklearn.metrics import classification_report

crflr = classification_report(y_test, y_predict)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

crfknn = classification_report(y_test, y_pred)



plt.figure(figsize=(10, 8))
sns.scatterplot(x = x['SepalLengthCm'],y = x['PetalLengthCm'], hue = y)








