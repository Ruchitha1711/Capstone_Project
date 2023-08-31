#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


file_path = "C:/Users/prave/Downloads/Playstore.csv"


# In[5]:


df = pd.read_csv("C:/Users/prave/Downloads/Playstore.csv",encoding ='ISO-8859-1')
print(df.head())


# In[6]:


#removing duplicates
df = df.drop_duplicates()
df  


# In[7]:


#calculating nulls
df.isnull().sum()


# In[8]:


import pandas as pd
df1 = df.fillna('NA')
df1.to_csv('filled_dataset.csv', index=False)


# In[9]:


df1.isnull().sum()


# In[10]:


df1.head()


# In[11]:


# Remove rows where the currency column has a value of "XXX"
df1 = df1[df1['Currency'] != 'XXX']

# Reset the index of the DataFrame
df1.reset_index(drop=True, inplace=True)

# Optionally, you can save the updated DataFrame back to a CSV file
df1.to_csv('updated_df1.csv', index=False)


# In[12]:


import re

# Define a regular expression pattern to match special characters
special_char_pattern = re.compile(r'[!@#$%^&*(),.?":{}|<>]')

# Filter out rows where 'App Name' column contains special characters
df1 = df1[~df1['App name'].str.contains(special_char_pattern)]

# Reset the index of the DataFrame
df1.reset_index(drop=True, inplace=True)

# Optionally, you can save the updated DataFrame back to a CSV file
df1.to_csv('cleaned_df1.csv', index=False)


# In[13]:


df1.info()


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt


# Convert 'NA' values to pd.NA
df1['Minimum Installs'] = df1['Minimum Installs'].replace('NA', pd.NA)
df1['Maximum Installs'] = df1['Maximum Installs'].replace('NA', pd.NA)

# Convert columns to numeric, treating 'NA' as NaN
df1['Minimum Installs'] = pd.to_numeric(df1['Minimum Installs'], errors='coerce')
df1['Maximum Installs'] = pd.to_numeric(df1['Maximum Installs'], errors='coerce')

# Filter out rows with missing values
df1_cleaned = df1.dropna(subset=['Rating', 'Minimum Installs', 'Maximum Installs'])

# Create a scatter plot for Rating vs Minimum Installs
plt.scatter(df1_cleaned['Rating'], df1_cleaned['Minimum Installs'], alpha=0.5)
plt.xlabel('Rating')
plt.ylabel('Minimum Installs')
plt.title('Rating vs Minimum Installs')
plt.show()

# Create a scatter plot for Rating vs Maximum Installs
plt.scatter(df1_cleaned['Rating'], df1_cleaned['Maximum Installs'], alpha=0.5)
plt.xlabel('Rating')
plt.ylabel('Maximum Installs')
plt.title('Rating vs Maximum Installs')
plt.show()


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Convert 'NA' values to pd.NA
df1['Minimum Installs'] = df1['Minimum Installs'].replace('NA', pd.NA)
df1['Maximum Installs'] = df1['Maximum Installs'].replace('NA', pd.NA)

# Convert columns to numeric, treating 'NA' as NaN
df1['Minimum Installs'] = pd.to_numeric(df1['Minimum Installs'], errors='coerce')
df1['Maximum Installs'] = pd.to_numeric(df1['Maximum Installs'], errors='coerce')

# Filter out rows with missing values
df1_cleaned = df1.dropna(subset=['Rating', 'Minimum Installs', 'Maximum Installs'])

# Define rating bins
rating_bins = np.arange(0, 5.1, 0.5)

# Create bar graph for Rating vs Minimum Installs
min_installs_bins = pd.cut(df1_cleaned['Rating'], bins=rating_bins)
min_installs_grouped = df1_cleaned.groupby(min_installs_bins)['Minimum Installs'].mean()
min_installs_grouped.plot(kind='bar')
plt.xlabel('Rating Range')
plt.ylabel('Average Minimum Installs')
plt.title('Bar Graph: Rating vs Average Minimum Installs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create bar graph for Rating vs Maximum Installs
max_installs_bins = pd.cut(df1_cleaned['Rating'], bins=rating_bins)
max_installs_grouped = df1_cleaned.groupby(max_installs_bins)['Maximum Installs'].mean()
max_installs_grouped.plot(kind='bar')
plt.xlabel('Rating Range')
plt.ylabel('Average Maximum Installs')
plt.title('Bar Graph: Rating vs Average Maximum Installs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
# Assuming you have a DataFrame named df1 with columns 'Size' and 'Price'
# Filter out rows with missing values
df1_cleaned = df1.dropna(subset=['Size', 'Price'])

# Create a scatter plot for Size vs Price
plt.scatter(df1_cleaned['Size'], df1_cleaned['Price'], alpha=0.5)
plt.xlabel('Size')
plt.ylabel('Price (Free/Paid)')
plt.title('Scatter Plot: Size vs Price')
plt.show()


# In[22]:


import pandas as pd



# Group the data by 'Category' and sum up the 'Maximum Installs' for each category
category_installs = df1.groupby('Category')['Maximum Installs'].sum()

# Find the category with the highest installs
max_installs_category = category_installs.idxmax()
max_installs_count = category_installs.max()

print(f"The category with the highest installs is '{max_installs_category}' with a total of {max_installs_count} installs.")


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt



# Group the data by 'Category' and sum up the 'Maximum Installs' for each category
category_installs = df1.groupby('Category')['Maximum Installs'].sum()

# Sort the categories by total installs in descending order
category_installs = category_installs.sort_values(ascending=False)

# Create a bar chart for Category vs Total Installs
plt.figure(figsize=(12, 8))
category_installs.plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Total Installs')
plt.title('Bar Chart: Category vs Total Installs')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[24]:


import pandas as pd


# Group the data by 'Category' and sum up the 'Maximum Installs' for each category
category_installs = df1.groupby('Category')['Maximum Installs'].sum()

# Sort the categories by total installs in descending order
category_installs = category_installs.sort_values(ascending=False)

# Convert the Series to a DataFrame for better formatting
category_installs_df = category_installs.reset_index()
category_installs_df.columns = ['Category', 'Total Installs']

# Display the DataFrame
print(category_installs_df)


# In[29]:


import pandas as pd

# Assuming you have a DataFrame named df1 with columns including 'price' and 'Maximum Installs'

# Group the data by 'Free' and calculate the sum of installs for each group
install_counts = df1.groupby('Free')['Maximum Installs'].sum()

# Print the install counts for free and paid apps
print("Total installs for free apps:", install_counts[True])
print("Total installs for paid apps:", install_counts[False])


# In[30]:


print(df1['Price'].dtype)


# In[31]:


import pandas as pd


# Filter free apps
free_apps = df1[df1['Free'] == False]

# Group by 'Category' and calculate install and rating statistics
category_stats = free_apps.groupby('Category').agg({'Maximum Installs': 'sum', 'Rating': 'mean'})

# Find the category with the highest number of installations
most_installed_category = category_stats['Maximum Installs'].idxmax()

# Get the average rating of the most installed category
average_rating_most_installed = category_stats.loc[most_installed_category, 'Rating']

# Print the results
print("Category with the most installations in paid app:", most_installed_category)
print("Average rating of the most installed category:", average_rating_most_installed)


# In[32]:


import pandas as pd


# Filter free apps
free_apps = df1[df1['Free'] == True]

# Group by 'Category' and calculate install and rating statistics
category_stats = free_apps.groupby('Category').agg({'Maximum Installs': 'sum', 'Rating': 'mean'})

# Find the category with the highest number of installations
most_installed_category = category_stats['Maximum Installs'].idxmax()

# Get the average rating of the most installed category
average_rating_most_installed = category_stats.loc[most_installed_category, 'Rating']

# Print the results
print("Category with the most installations in free apps:", most_installed_category)
print("Average rating of the most installed category:", average_rating_most_installed)


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns


# Calculate the distribution of unique categories
category_distribution = df1['Category'].value_counts()

# Create a bar chart for the category distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=category_distribution.index, y=category_distribution.values)
plt.xticks(rotation=90)
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Distribution of App Categories')
plt.tight_layout()
plt.show()


# In[35]:


import pandas as pd



# Calculate the distribution of apps category-wise
category_distribution = df['Category'].value_counts()

# Display the distribution
print(category_distribution)


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt


# Group by Category and Free columns, then count the number of occurrences
category_free_distribution = df1.groupby(['Category', 'Free'])['App name'].count()

# Unstack the Free column to create a pivot table for visualization
category_free_distribution = category_free_distribution.unstack(level=1, fill_value=0)

# Plot the bar chart
category_free_distribution.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("App Distribution by Category and Free/Paid")
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.legend(title='Free')
plt.show()


# In[41]:


import pandas as pd


# Count the occurrences of each category and select the top 6
top_categories = df['Category'].value_counts().head(6)

print(top_categories)


# In[42]:


import pandas as pd



# Group by category and calculate the average rating and average number of installs
category_stats = df1.groupby('Category').agg({'Rating': 'mean', 'Maximum Installs': 'mean'})

# Sort categories by average rating in descending order
best_rating_categories = category_stats.sort_values(by='Rating', ascending=False)

# Sort categories by average installs in descending order
best_installs_categories = category_stats.sort_values(by='Maximum Installs', ascending=False)

print("Best performing categories by average rating:")
print(best_rating_categories.head())

print("\nBest performing categories by average installs:")
print(best_installs_categories.head())


# In[ ]:




