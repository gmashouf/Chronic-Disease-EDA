# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('U.S._Chronic_Disease_Indicators.csv')


#############################
# sub-dataset based on topic
#############################
dataset_Alcohol = dataset.loc[(dataset["Topic"]=="Alcohol")]
dataset_Arthritis = dataset.loc[(dataset["Topic"]=="Arthritis")]
dataset_Asthma = dataset.loc[(dataset["Topic"]=="Asthma")]

###################
# plot variable by state
###################

ax = sns.boxplot(x="LocationAbbr", y="DataValueAlt", data=dataset_Arthritis)
ax = sns.boxplot(x="LocationAbbr", y="DataValueAlt", data=dataset_Asthma)
ax = sns.boxplot(x="LocationAbbr", y="DataValueAlt", data=dataset_Alcohol)
show.plt



#dataset_Alcohol.boxplot(column="DataValueAlt",by="LocationAbbr" )


X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 10].values  # Data Value Coloumn

# converts objects to float 
y = pd.to_numeric(y, errors ='coerce')
y = np.array(y).reshape((1, -1))

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(y)
y = imputer.transform(y)
y = np.array(y).reshape((-1, 1))  # returns y to its orginal size


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

######################
# defing a class for MultiColumnLabelEncoder
"""" 
https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
""""
######################



# removes the priority of dummy variables
# onehotencoder = OneHotEncoder(categorical_features = [2])
# X = onehotencoder.fit_transform(X).toarray()



# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


