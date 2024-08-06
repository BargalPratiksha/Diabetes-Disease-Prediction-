#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np


# # visualizing the data

# In[5]:


df=pd.read_csv(r"C:\Users\Pratiksha Bargal\Downloads\diabetes.csv")


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df["Glucose"]=df["Glucose"].replace(0,np.nan)
df["BloodPressure"]=df["BloodPressure"].replace(0,np.nan)
df["SkinThickness"]=df["SkinThickness"].replace(0,np.nan)
df["Insulin"]=df["Insulin"].replace(0,np.nan)
df["BMI"]=df["BMI"].replace(0,np.nan)


# In[11]:


df.head()


# In[12]:


df.describe()


# # Data Cleaning

# In[13]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
imputed_array = imputer.fit_transform(df)
imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
df = imputed_df 


# # Data in a pictorial format

# In[14]:


plt.figure(figsize=(5,5))

plt.pie(df['Outcome'].value_counts(),labels=['Non-diabetic','Diabetic'],radius=1,
        autopct='%1.1f%%',labeldistance=1.15)

plt.legend(title = 'Outcome:',loc='upper right', bbox_to_anchor=(1.6,1))
plt.show()


# In[15]:


plt.figure(figsize=(14, 8))
sns.boxplot(df)
plt.title("the columns before handling outliers")
plt.show()


# In[16]:


for col in df:
    # Calculate IQR and identify potential outliers
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Clip the values to the lower and upper bounds
    clipped_values = np.clip(df[col], lower_bound, upper_bound)

    # Assign the clipped values back to the DataFrame
    df[col] = clipped_values

plt.figure(figsize=(14, 6))
sns.boxplot(data=df)
plt.title("the columns after handling outliers")
plt.show()


# In[17]:


df.hist(bins=50, figsize=(20,15));


# In[18]:


#correlation map
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);


# In[19]:


sns.scatterplot(x='Glucose', y='Outcome', data=df)
plt.title('Scatter Plot between Glucose and Outcome')
plt.xlabel('Glucose')
plt.ylabel('Outcome')
plt.show()


# In[20]:


sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title('Boxplot of Glucose by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Glucose')
plt.show()


# In[21]:


# Glucose Distribution
hist = px.histogram(data_frame=df, x='Glucose', color='Outcome', title="Glucose Distribution", height=500,color_discrete_map={0: 'black', 1: 'orange'})
hist.update_layout({'title':{'x':0.5}})
hist.show();


# In[22]:


# Insulin Distribution
import plotly.express as px
hist = px.histogram(data_frame=df, x="Insulin", color='Outcome', title="Insulin Distribution", height=500,color_discrete_map={0: 'black', 1: 'orange'})
hist.update_layout({'title':{'x':0.5}})
hist.show();


# In[23]:


# BMI Distribution
import plotly.express as px
hist = px.histogram(data_frame=df, x="BMI", color='Outcome', title="BMI Distribution", height=500,color_discrete_map={0: 'black', 1: 'orange'})
hist.update_layout({'title':{'x':0.5}})
hist.show();


# # modeling

# In[24]:


X = df.drop("Outcome", axis = 1)
y = df["Outcome"]


# # Training and testing

# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = .2, random_state=0)


# # Accuracy of a model

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(kernel='linear', C=1),
    KNeighborsClassifier(n_neighbors=3),
    LogisticRegression(random_state=42, max_iter=1000)  # Addressed convergence warning
]

# Initialize an empty DataFrame to store results
results = pd.DataFrame(columns=["Model", "Train Accuracy", "Test Accuracy"])

# Train and evaluate each model, storing results in the DataFrame
for model in models:
    name = model.__class__.__name__  # Access model name
    model.fit(X_train, y_train)  # Fit the model directly on the original training set
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy using original y_train
    test_accuracy = accuracy_score(y_test, y_test_pred)
    results.loc[len(results.index)] = {"Model": name, "Train Accuracy": train_accuracy, "Test Accuracy": test_accuracy}
    print("{} Accuracy: {:.2f}%".format(name, test_accuracy * 100))
    print("{} Classification Report:\n{}".format(name, classification_report(y_test, y_test_pred)))
    print("\n" + "="*50 + "\n")

# Print the results DataFrame
print(results)


# In[ ]:




