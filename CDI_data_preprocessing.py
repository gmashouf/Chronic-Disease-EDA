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
dataset_Cancer = dataset.loc[(dataset["Topic"]=="Cancer")]
dataset_Diabetes = dataset.loc[(dataset["Topic"]=="Diabetes")]
dataset_Overarching_Conditions = dataset.loc[(dataset["Topic"]=="Overarching Conditions")]
dataset_Tobacco = dataset.loc[(dataset["Topic"]=="Tobacco")]
dataset_Chronic_Kidney_Disease = dataset.loc[(dataset["Topic"]=="Chronic Kidney Disease")]
dataset_Chronic_Obstructive_Pulmonary_Disease = dataset.loc[(dataset["Topic"]=="Chronic Obstructive Pulmonary Disease")]
dataset_Cardiovascular_Disease = dataset.loc[(dataset["Topic"]=="Cardiovascular Disease")]
dataset_Immunization = dataset.loc[(dataset["Topic"]=="Immunization")]
dataset_Mental_Health = dataset.loc[(dataset["Topic"]=="Mental Health")]
dataset_Nutrition = dataset.loc[(dataset["Topic"]=="Nutrition")]
dataset_Physical_Activity = dataset.loc[(dataset["Topic"]=="Physical Activity")]
dataset_and_Weight_Status = dataset.loc[(dataset["Topic"]=="and Weight Status")]
dataset_Reproductive_Health = dataset.loc[(dataset["Topic"]=="Reproductive Health")]
dataset_Oral_Health = dataset.loc[(dataset["Topic"]=="Oral Health")]
dataset_Older_Adults = dataset.loc[(dataset["Topic"]=="Older Adults")]
dataset_Disability = dataset.loc[(dataset["Topic"]=="Disability")]

#######################################
# sub_datasets with specific condition
#######################################
dataset_Cancer_WA = dataset_Cancer.loc[(dataset_Cancer["LocationAbbr"] == "WA") &\
                                       (dataset_Cancer["Stratification1"] != "Overall")]
dataset_Cancer_all = dataset_Cancer.loc[(dataset_Cancer["DataValueType"] == "Age-adjusted Prevalence")]
###################################################
#plot cancer for all state
###################################################
ax = sns.factorplot(x="LocationAbbr", y="DataValueAlt", \
               data=dataset_Cancer_all[(dataset_Cancer_all.YearStart.notnull()) & (dataset_Cancer_all.DataValueAlt.notnull())],\
               hue = "YearStart", row = "Question", col = "DataValueType" ,\
               kind="bar", size = 40, aspect = 0.9, legend = True)
plt.setp(ax.get_xticklabels(), rotation=45)
sns.set(font_scale = 1)



##################################################
#plot cancer data set for WA state
###################################################

sns.factorplot(x="YearStart", y="DataValueAlt",\
               data=dataset_Cancer_WA[dataset_Cancer_WA.DataValueAlt.notnull()],\
               hue = "Stratification1", row = "Question", col = "DataValueType" ,\
               kind="bar", size = 20, aspect = 0.8, legend = True, margin_titles = True)
sns.set(font_scale = 1)


fig, ax = plt.subplots(1, 1, figsize=(50,50))
plt.setp(ax.get_xticklabels(), rotation=45)

ax = sns.barplot(x="YearStart", y="DataValueAlt", data=dataset_Cancer_WA, hue = "Stratification1")
ax = sns.barplot(x="YearStart", y="DataValueAlt", data=dataset_Cancer_WA, hue = "DataValueType")
show.plt


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


