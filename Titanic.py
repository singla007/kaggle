import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("train.csv")
print(dataset.columns)
features = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']
dataset = dataset.dropna(subset=['Embarked'])
X = dataset[features].values
y = dataset['Survived'].values


dataset[features].describe()

print(X[:,2])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:,2:3])
print(X)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,1] = labelencoder.fit_transform(X[:,1])

labelencoder2 = LabelEncoder()
X[:,6] = labelencoder2.fit_transform(X[:,6])

print(X)
onehotencoder = OneHotEncoder(categorical_features = [0,1,6])
X = onehotencoder.fit_transform(X).toarray()
print(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.ensemble import RandomForestRegressor
rg = RandomForestRegressor(n_estimators = 4, random_state = 0)
rg.fit(X,y)



"""_______________output_____________________________________"""



dataset = pd.read_csv("test.csv")
print(dataset.columns)
features = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']
dataset = dataset.dropna(subset=['Embarked'])
X= dataset[features].values



dataset[features].describe()



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:,2:3])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,5:6])
X[:,5:6] = imputer.transform(X[:,5:6])
dt = pd.DataFrame(X)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,1] = labelencoder.fit_transform(X[:,1])

labelencoder2 = LabelEncoder()
X[:,6] = labelencoder2.fit_transform(X[:,6])


onehotencoder = OneHotEncoder(categorical_features = [0,1,6])
X = onehotencoder.fit_transform(X).toarray()


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

y_pred = rg.predict(X)

print(len(y_pred))

output = []
for i in y_pred:
	if i <= 0.5 :
		output.append(0)
	else:
		output.append(1)

df1 = pd.DataFrame(output)
df1.to_csv('out1.csv',index = False)


