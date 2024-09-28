#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# In[17]:


# Sample of iris dataset
setosa = np.array([5.1, 4.9, 4.7, 4.6, 5, 5.4, 4.6, 5, 4.4, 4.9])
versicolor = np.array([7, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2])
virginica = np.array([6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2])


# In[28]:


# Calculate the averages
averages = {
    'Setosa': np.mean(setosa),
    'Versicolor': np.mean(versicolor),
    'Virginica': np.mean(virginica)
}

# Prepare data for plotting 
# Organize the species names and their corresponding average sepal lengths into lists for plotting.
species = list(averages.keys())
average_values = list(averages.values())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(species, average_values, marker='o', linestyle='-', color='b')

# Adding titles and labels
plt.title('Average Sepal Lengths of Iris Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.ylim(4, 8)  # Set y-axis limits for better visualization

# Show the plot
plt.grid(True)
plt.show()


# In[9]:


# Perform ANOVA
f_statistic, p_value = stats.f_oneway(setosa, versicolor, virginica)

print(f"F-Statistic: {f_statistic}")
print(f"P-Value: {p_value}")


# In[29]:


import numpy as np

# sample of iris dataset
setosa = np.array([5.1, 4.9, 4.7, 4.6, 5, 5.4, 4.6, 5, 4.4, 4.9])
versicolor = np.array([7, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2])
virginica = np.array([6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2])

# Combine all data into a single array
all_data = np.concatenate([setosa, versicolor, virginica])
print(all_data)

# Calculate group means
mean_setosa = np.mean(setosa)
mean_versicolor = np.mean(versicolor)
mean_virginica = np.mean(virginica)

# Calculate overall mean
overall_mean = np.mean(all_data)

# Number of observations per group
n_setosa = len(setosa)
n_versicolor = len(versicolor)
n_virginica = len(virginica)

# Calculate SSB (Between-group sum of squares)
SSB = (n_setosa * (mean_setosa - overall_mean) ** 2 +
       n_versicolor * (mean_versicolor - overall_mean) ** 2 +
       n_virginica * (mean_virginica - overall_mean) ** 2)

# Calculate SSW (Within-group sum of squares)
SSW = (np.sum((setosa - mean_setosa) ** 2) +
       np.sum((versicolor - mean_versicolor) ** 2) +
       np.sum((virginica - mean_virginica) ** 2))

# Calculate SST (Total sum of squares)
SST = np.sum((all_data - overall_mean) ** 2)

# Calculate degrees of freedom
df_between = 2  # k - 1, where k is the number of groups
df_within = len(all_data) - 3  # n - k, where n is the total number of observations

# Calculate mean squares
MSB = SSB / df_between
MSW = SSW / df_within

# Calculate F-statistic
F = MSB / MSW

# Print results
print(f"Mean Setosa: {mean_setosa}")
print(f"Mean Versicolor: {mean_versicolor}")
print(f"Mean Virginica: {mean_virginica}")
print(f"Overall Mean: {overall_mean}")
print(f"SSB (Between-group sum of squares): {SSB}")
print(f"SSW (Within-group sum of squares): {SSW}")
print(f"SST (Total sum of squares): {SST}")
print(f"df_between: {df_between}")
print(f"df_within: {df_within}")
print(f"MSB (Mean square between groups): {MSB}")
print(f"MSW (Mean square within groups): {MSW}")
print(f"F-statistic: {F}")
print(f"P-Value: {p_value}")


# Interpretation: As the p-value is less than the significance level (typically 0.05), we reject the null hypothesis.
# Conclusion: The ANOVA test shows that the mean sepal lengths for at least one of the species are significantly different from the others, meaning that not all group means are equal. This suggests a statistically significant difference in the average sepal lengths among the species.

# Practical Implications:
# Biological Significance: Different species of iris flowers have different average sepal lengths. This might be related to their evolutionary adaptations, ecological niches, or other biological factors.
# Classification: Knowing that sepal lengths differ significantly among species can help in classifying and identifying species based on their sepal measurements.
# By identifying and confirming these differences, researchers and scientists can better understand the characteristics and variations within and between species.
# 
