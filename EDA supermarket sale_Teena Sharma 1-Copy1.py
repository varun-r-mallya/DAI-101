#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[3]:


os.chdir("C:/Users/teena/OneDrive/Documents/Data analytics/Assignment 1")


# >Reading the csv file 

# In[32]:


data = pd.read_csv('supermarket_sales - Sheet1.csv',parse_dates=['Date'])


# >Ignoring warnings

# In[33]:


import warnings


# In[34]:


warnings.filterwarnings('ignore')


# >Displaying first 5 data of the data sheet

# In[35]:


data.head()


# In[36]:


data.tail()


# In[37]:


df.shape


# In[39]:


print("Number of rows", df.shape[0])
print("Number of columns", df.shape[1])


# In[42]:


data.isnull().sum()


# In[41]:


data.info()


# In[17]:


data.describe()


# In[ ]:


# Dividing the columns into categories and numericals data types
# Initialize empty lists: cat for categorical columns and num for numerical columns.
# Iterate over each column in the DataFrame:
# Check the number of unique values in the column:
# Classify the column based on the number of unique values:
# 1. If the column has more than 10 unique values, it is considered numerical and appended to the num list.
# 2. If the column has 10 or fewer unique values, it is considered categorical and appended to the cat list.


# In[44]:


cat=[]
num=[]
for column in data.columns:
    if data[column].nunique()>10:
        num.append(column)
    else:
        cat.append(column)
        
        


# In[45]:


cat


# In[46]:


num


# ## 1.Univariate analysis

# ### Categorical 

# >Displaying the numbers of different branches using pie chart

# In[21]:


data['Branch'].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title('Number of branches')


# Here we can see that the percentage of Branch A is greater than Branch B and C

# >Displaying number of different branches using barchart

# In[22]:


data['Branch'].value_counts()


# In[23]:


data['Branch'].value_counts().plot(kind="bar")
plt.title('Mode of branches')


# Plot 1: In the Bar chart, we can also see that number of Branch A is greater than that of number of branch B and C.

# In[24]:


data['Payment'].value_counts()


# In[25]:


data['Payment'].value_counts().plot(kind="bar")
plt.title('Mode of payment')


# Plot 2: Mode of payment by Ewallet and cash is almost same and by credit card is less. Number of payments by E-wallets is 345 and number of payments by credit card is 311. That shows that customers prefer to pay through cash and E-wallets as compared to credit card.

# ### Numerical

# In[26]:


num


# >Plotting the distribution of Ratings

# In[27]:


sns.distplot(data['Rating'])
plt.title('Distribution of rating')


# Plot 3: By the distribution graph of Rating, we can find that the ratings are normally distributed and the skewness is also almost equal to zero i.e. the distribution is symmetrical from both side.

# In[28]:


data['Rating'].skew()


# >Plotting the distribution of cost of goods sold (cogs)

# In[29]:


sns.distplot(data['cogs'])
plt.title('Distribution of cogs')


# P4: Here we can see that data is right skewed. In other words, the majority of the data is concentrated on the left side of the distribution, with a few high values on the right side. The number of low cogs is more than that of number of high cogs. We can find the probability of the particular cogs by KDE(Kernel density estimation). For example- probability of cogs 400 is approximately 0.1 %

# In[30]:


data['cogs'].skew()


# In[31]:


data['cogs'].describe()


# In[32]:


print("Median of cost of goods(cogs)- ",data['cogs'].median())


# We have found the mean, median, std, min and max of cogs

# In[33]:


sns.boxplot(data['cogs'])
plt.title('Box Plot of cog')


# P5: In this box plot, we can se the minimun and maximum value of cogs. the middle line in the blue box denotes the median of the cogs i.e 241. The lower bound of the box denotes the 1st quartile i.e the 25% of value lies between minimum and 1st quartile value. The upper bound of box denotes 3rd quartile. We have outliers over upper bound (above 1.5 times IQR(inter-quartile ratio)) which denotes extreme values, or the presence of an entirely different underlying process.

# ## 2. Bivariate analysis

# ### Numerical-Numerical

# >scatter plot of gross income and cogs

# In[30]:


sns.scatterplot(data = data,x="gross income",y="cogs")

Plot 1: By the scatterplot betweeen gross income and cogs, we can see that there is direct relationship between the gross income and cogs. As the cogs is increased, the gross income on that sale is also get increased.
# ### Categorical-Numerical

# >Bar plot of Branch with gross income

# In[48]:


sns.barplot(data=data,x="Branch",y="gross income")


# P1: The gross income is highest for branch C and lowest for branch A.
# the hue parameter is used to introduce a third variable that will produce bars with different colors within each category of the x variable. This helps in visualizing the relationship between x and y variables, grouped by the hue variable.
# In[47]:


sns.barplot(data=data,x="Branch",y="gross income",hue="City")

# P2: Gross income for Branch C is highest (in city Naypyitaw) and branch A is lowest in city Yangon.
# #### Relationship with Gender and gross income

# In[33]:


sns.boxplot(data=data, x="Gender", y="gross income")


#  P3: By this, we can see that the gross income from both male and female consumers are equal. But female consumers spend a little bit more as compared to male consumers.
# 
# 

# In[34]:


sns.boxplot(data=data, x="Gender", y="gross income", hue="Customer type")


# #### Product line which generates most gross income

# In[35]:


sns.barplot(data=data,x="Product line", y="gross income")
plt.title('Product line which generate most gross income')
plt.xticks(rotation=80)
plt.show()

P4: The Home and lifestyle product line generates highest gross income among all the product line and Fashion accessories generates the lowest.
# #### Product line vs Unit price

# In[36]:


sns.barplot(data=data,x="Product line", y="Unit price")
plt.title('Unit price vs Product line')
plt.xticks(rotation=80)
plt.show()

P5: The highest unit price is of fashion accessories and sports and travel product line.
# ### Categorical-Categorical 

# >Different payment methods used by customer citywise

# In[37]:


pd.crosstab(data['City'],data['Payment'])


# In[38]:


sns.heatmap(pd.crosstab(data['City'],data['Payment']))

Plot 1: The consumers from city Naypyitaw pay more in cash and least by credit card. The consumers of City Mandalay uses almost all methods of payment equally.
# #### Which product line is produced in the highest quantity

# In[39]:


(data.groupby('Product line').sum()['Quantity']).plot(kind="bar")

P2: Here we can see that electronic accessories has beeen produced in highest quantity
# #### Display daily sales by day of the week

# In[40]:


data.columns


# In[41]:


dw_mapping={
    0:'Mon',
    1:'Tue',
    2:'Wed',
    3:'Thur',
    4:'Fri',
    5:'Sat',
    6:'Sun'
}


# In[42]:


data['day_of_week']=data['Date'].dt.dayofweek.map(dw_mapping)


# In[43]:


data['day_of_week'].value_counts().plot(kind="bar")

P3: Saturday has the highest number of daily sales and Mon
day has the lowest daily sales.
# # 3. Multivariate Analysis

# In[54]:


sns.barplot(data=data,x="Branch",y="gross income",hue="City")

P1: By the bar plot, we see can see the different branches in different cities. Branch B which is in Naypyitaw has the highest gross income and Branch A which is in the city Yangon has the lowest gross income.As we have seen earlier, number of branch A is also more than branch B and C. This means that we should have take some important measures to increase the gross income in the City Yangon.
# In[55]:


sns.boxplot(data=data, x="Gender", y="gross income", hue="Customer type")

P2: By this, we see that female member consumer spend more than normal male or female consumer.
# # Conclusion:
# 

# 1)- We have to make our focus on city Yangon as it has the most number of branches but still generates low gross income as compared to others.
# 
# 2)- We have to spend less on fashion accessories as its unit price is high but generates lowest gross income.
# 
# 3)- We have to take care of female member consumer as they are important part of total gross income of supermarket.
# 
# 4)- Try to convert normal consumer to member consumer.
# 
# 5)- Be ready for the high sale in Saturday and try to attract more consumer on Sunday and Monday.
