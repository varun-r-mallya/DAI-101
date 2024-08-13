#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import GridSearchCV
import os
import seaborn as sns
from scipy import stats


# In[2]:


os.chdir ("C:/Users/teena/OneDrive/Documents/Data Analytics/Jupyter Work")


# In[3]:


#  Load the dataset into a pandas DataFrame.
df = pd.read_csv('titanic.csv')
print (df.shape)


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


print(df.describe())


# In[9]:


df.dtypes


# In[10]:


# check duplicate values in each row
df.duplicated().sum


# In[11]:


# value_counts to check uniques values in each column (for column Sex uniques values 577 for maile and 314 for female)
df.Sex.value_counts()


# In[12]:


df.Survived.value_counts()


# In[13]:


df.Survived.value_counts().plot(kind='bar')


# In[14]:


#rename column
df.rename(columns={'Pclass':'Passenger class'})
df.head(2)


# In[15]:


# # The isnull() method returns a DataFrame object where all the values are replaced with a Boolean value True for NULL values, and otherwise False
df.isnull()


# In[16]:


# null check(missing values in each column)
# df.isnull().sum returns the number of missing values in the dataset.
df.isnull().sum()


# In[17]:


# Handle missing values

# Option 1: Drop missing values (drop rows with any missing values)
# df.dropna(inplace=True) [Dropna drops or discards all the rows with missing values in the dataset]
# By default, the dropna() method returns a new DataFrame, and will not change the original.

# If you want to change the original DataFrame, use the inplace = True argument:

# Option 2: Fill missing values with mean/median/mode 
# df["Age"].fillna(df["Age"].mean(), inplace=True)
# df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


# In[18]:


df['Age'].mean()


# In[19]:


# The groupby method in pandas is used to group data by specific columns and then apply aggregate functions, such as mean(), to each group.
df.groupby(by='Pclass')['Age'].mean()


# In[20]:


df.groupby(by='Pclass')['Age'].mean().plot.bar()


# In[21]:


# DataFrame df should now properly fill missing ages based on the passenger class
# Define the function m_Age. The m_Age function fills the missing 'Age' values based on the 'Pclass' of the passenger. 
def m_Age(c):
    Age = c[0]
    Pclass = c[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 25
    else:
        return Age
    
# Apply the function to fill missing 'Age' values. The apply function iterates over each row, and m_Age checks if the 'Age' is missing (NaN). If 'Age' is missing, it assigns an average age based on 'Pclass'; otherwise, it keeps the original 'Age'.
df['Age']=df[['Age', 'Pclass']].apply(m_Age, axis=1)
print(df)


# In[22]:


df.isnull().sum()


# In[23]:


df.drop('Cabin', axis=1, inplace=True)


# In[24]:


df.isnull().sum()


# In[25]:


sns.heatmap(df.isnull())


# In[26]:


df.groupby(by='Embarked')['Age'].mean().plot.bar()


# In[27]:


df.groupby(by='Embarked')['Age'].mean().plot.bar()


# In[28]:


# converting catagorial to numeric
#Use pd.get_dummies() to convert any column to dummy variables.
#Join the resulting dummy variables back to the original DataFrame.


# In[29]:


# Create dummy variables for the 'Sex' and 'Embark' columns using pd.get_dummies()


# In[30]:


Sex=pd.get_dummies(df['Sex'], drop_first=True)


# In[31]:


Sex


# In[32]:


Embarked=pd.get_dummies(df['Embarked'],drop_first=True)


# In[33]:


Embarked


# In[34]:


# Concatenate these dummy variables with the original DataFrame using pd.concat().
df = pd.concat([df, Sex, Embarked], axis=1)


# In[35]:


df.head()


# In[36]:


# To drop or delete unnecessary columns
df.drop(['Name', 'Ticket'], axis=1, inplace=True)


# In[37]:


df.head()


# In[38]:


# Column to check for outliers (boxplot method)   
# dropna with df[column] means it's excluding NaN values
column = 'Age'

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df[column].dropna())
plt.show()


# In[39]:


# Column to check for outliers
column = 'Age'

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(df[column].dropna(), bins=10, edgecolor='black')
plt.title(f'Histogram of {column}')
plt.xlabel(column)
plt.ylabel('Frequency')
plt.show()


# In[40]:


# Z-score method to detect outliers

column = 'Age'

# Convert the column to a numpy array, excluding NaN values
column_np = df[column].dropna().to_numpy()

# Calculate mean and standard deviation
mean = np.mean(column_np)
std_dev = np.std(column_np)

# Print mean and standard deviation
print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")

# Calculate Z-scores
z_scores = stats.zscore(column_np)

# Print Z-scores
print(f"Z-scores: {column_np}")

# Identify outliers (using a threshold of 3)
threshold = 3
outliers = column_np[np.abs(z_scores) > threshold]

print("Outliers using Z-score method:")
print(outliers)


# In[41]:


# Z-score method to detect outliers (direct and short way)
# Column to check for outliers
column = 'Age'

# Calculate Z-scores
z_scores = np.abs(stats.zscore(df[column].dropna()))

# Set a threshold (commonly 3)
threshold = 3

# Identify outliers
outliers = df[column][(z_scores > threshold)]

print("Outliers using Z-score method:")
print(outliers)


# In[42]:


# Filter out the outliers
df_cleaned_zscore = df[(np.abs(stats.zscore(df[column].dropna())) <= threshold) | df[column].isna()]

print("Dataset after removing outliers using Z-score method:")
print(df_cleaned_zscore[column])


# In[43]:


# Detect outliers using IQR method


# Column to check for outliers
column = 'Age'

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Print Q1, Q3, and IQR values
print(f"Q1 (25th percentile) of {column}: {Q1}")
print(f"Q3 (75th percentile) of {column}: {Q3}")
print(f"IQR (Interquartile Range) of {column}: {IQR}")


# Identify outliers
outliers_iqr = df[column][((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

print("Outliers using IQR method:")
print(outliers_iqr)


# In[44]:


# Filter out the outliers

df_cleaned_iqr = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))) | df[column].isna()]

print("Dataset after removing outliers using IQR method:")
print(df_cleaned_iqr[column])


# In[ ]:





# In[ ]:





# In[ ]:




