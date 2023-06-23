#!/usr/bin/env python
# coding: utf-8

# ### Importing Dependencies 

# In[26]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[27]:


import os
import seaborn as sns

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print('Dependencies Imported')


# In[28]:


data = pd.read_csv("Coursera.csv") #reading the dataset
data 


# In[39]:


fig, ax = plt.subplots(figsize=(12,10))
sns.countplot(data['Course Rating'], ax=ax)
plt.title('Course Rating Chart')
# plt.savefig('stars.png')
plt.show()


# ### Basic Data Analysis 

# In[29]:


data.shape #263 courses and 6 columns with different attributes


# In[30]:


data.info() #checking overall courses with rating and total courses 


# In[31]:


data.isnull().sum() #checking if there is any missing value 


# In[32]:


data['Difficulty Level'].value_counts() #checking the difficulty level of courses 


# In[33]:


data['Course Rating'].value_counts() #Checking the rating of courses 


# In[34]:


data['University'].value_counts() #checking the universities in dataset


# In[35]:


data['Course Name']


# #### Selecting the Specific Column for Data Processing

# In[36]:


data = data[['Course Name','Difficulty Level','Course Description','Skills']]
data.head(5)


# In[37]:


def plotPerColumnDistribution(data, g, h): 

plotPerColumnDistribution(data, 10, 5)

plotCorrelationMatrix(df2, 8)


# ### Data Preprocessing 

# #### Removing the Useless Columns and Spaces

# In[ ]:


# Removing spaces between the words (Lambda funtions can be used as well)
# Removing the useless columns and spaces to clean the data

data['Course Name'] = data['Course Name'].str.replace(' ',',')
data['Course Name'] = data['Course Name'].str.replace(',,',',')
data['Course Name'] = data['Course Name'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace(' ',',')
data['Course Description'] = data['Course Description'].str.replace(',,',',')
data['Course Description'] = data['Course Description'].str.replace('_','')
data['Course Description'] = data['Course Description'].str.replace(':','')
data['Course Description'] = data['Course Description'].str.replace('(','')
data['Course Description'] = data['Course Description'].str.replace(')','')

#removing paranthesis from skills columns 

data['Skills'] = data['Skills'].str.replace('(','')
data['Skills'] = data['Skills'].str.replace(')','')


# In[ ]:


d = data
d


# In[ ]:


# Null/Removal data checking
print("Are there any missing values in the dataset ?",data.isna().values.any())


# In[ ]:


# complete summary of dataset
data.describe().T


# #### Data Visualization

# In[ ]:


mag = ['Beginner','Advanced','Intermediate','Conversant','Not Calibrated']
  
data = [1444,1005,837,186,50]
explode = (0.1,0.0,0.0,0.0,0.0) 
  
# Creating color parameters 
colors = ("lightblue","crimson","yellow","green","violet") 
  
# Wedge properties 
wp = { 'linewidth' : 1, 'edgecolor' : "white" } 
  
# Creating autocpt arguments 
def func(pct, allvalues): 
    absolute = int(pct / 100.*np.sum(allvalues)) 
    return "{:.1f}%\n({:d} g)".format(pct, absolute) 
  
# Creating plot 
fig, ax = plt.subplots(figsize =(15, 10)) 
wedges, texts, autotexts = ax.pie(data,  
                                  autopct = lambda pct: func(pct, data), 
                                  explode = explode,  
                                  labels = mag, 
                                  shadow = True, 
                                  colors = colors, 
                                  startangle = 90, 
                                  wedgeprops = wp, 
                                  textprops = dict(color ="black")) 
  
# Adding legend 
ax.legend(wedges, mag, 
          title ="Values", 
          loc ="center left", 
          bbox_to_anchor =(1, 0, 0.5, 1)) 
  
plt.setp(autotexts, size = 10, weight ="bold") 
ax.set_title("Payment type of course\n",size=19) 
  
# show plot 

plt.show()


# In[ ]:


# plt.figure(figsize=(18,7))
# sns.countplot(data,x='Course Rating',palette='plasma')
# plt.xlabel('Course Ratings',fontsize='16',color='blue')
# plt.ylabel('Number of courses',fontsize='16',color='blue')
# plt.xticks(fontsize='14',color='green')
# plt.yticks(fontsize='14',color='red')
# plt.title("Count of course types\n",fontsize=24,fontweight='bold',color='indigo')


# #### Filtering the Dataset

# In[ ]:


# from mlxtend.preprocessing import minmax_scaling

# # generate 1000 data points randomly drawn from an exponential distribution
# data = np.random.exponential(size = 1000)

# # mix-max scale the data between 0 and 1
# scaled_data = minmax_scaling(data, columns = [0])

# # plot both together to compare
# fig, ax=plt.subplots(1,2)
# sns.distplot(original_data, ax=ax[0])
# ax[0].set_title("Original Data")
# sns.distplot(scaled_data, ax=ax[1])
# ax[1].set_title("Scaled data")


# In[ ]:


# ## Lower Casing
# data["reviews_list"] = data["reviews_list"].str.lower()

# ## Removal of Puctuations
# import string
# PUNCT_TO_REMOVE = string.punctuation
# def remove_punctuation(text):
#     """custom function to remove the punctuation"""
#     return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

# data["reviews_list"] = data["reviews_list"].apply(lambda text: remove_punctuation(text))

# ## Removal of Stopwords
# from nltk.corpus import stopwords
# STOPWORDS = set(stopwords.words('english'))
# def remove_stopwords(text):
#     """custom function to remove the stopwords"""
#     return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# data["reviews_list"] = data["reviews_list"].apply(lambda text: remove_stopwords(text))

# ## Removal of URLS
# def remove_urls(text):
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# data["reviews_list"] = data["reviews_list"].apply(lambda text: remove_urls(text))

# data[['reviews_list', 'cuisines']].sample(5)


# In[ ]:


#!pip install seaborn==0.11.1
#!pip install wordcloud==1.8.1


# In[ ]:


#x = pd.DataFrame(data).iloc[2:, : ].reset_index().rename(columns={'index': 'Skills', 0: 'count'}).sort_values(by='count', ascending=False)
#x


# In[ ]:


# sns.barplot(data=x, x="Skills", y="count")
# plt.xticks(rotation=90)
# plt.show()


# In[ ]:


# sns.distplot(data['Course Rating'])
# plt.show()


# In[ ]:


#pip instal cross_validation


# In[ ]:


# from sklearn.model_selection import train_test_split
     
# X = data.drop("output", axis=1) #data[["contrast", "entropy"]]
# # y = data[["output"]]
# # X_train , X_test , y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.20)



# In[ ]:


from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np


# labels = pd.read_csv('Labels.csv')
# features = pd.read_csv('Features.csv')

def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
    distances = []
    targets = []
    for i in range(len(X_train)):
		# first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# add it to list of distances
        distances.append([distance, i])
    distances = sorted(distances)
	# make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index][0])
    return targets

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))
    return predictions
        

accuracy_for_eachK = []
count_arr = []
for K in [7,9]:
    accuracy_for_each_fold = []
    print ("K Values:",K)
#     for b in [5,10]:
#         X_train, X_test, y_train, y_test = train_test_split(np.array(features),np.array(labels),test_size=0.1)
#         predictions,count = [],0
#         print ("B Values:",b)
#         kNearestNeighbor(X_train, y_train, X_test, predictions, K)
#         for i in xrange(len(predictions)):
#             if y_test[i] in predictions[i]:
#                 count+=1
#         count_arr.append(count);
#         print (count)
#         print (count*1.0/len(predictions)*100.0)
#         accuracy_for_each_fold.append((count*1.0/len(predictions))*100.0)
#     accuracy_for_eachK.append(sum(accuracy_for_each_fold))
#     print (accuracy_for_eachK)


# In[ ]:


# from sklearn.model_selection import train_test_split
# X = data.drop("output", axis=1) #data[["contrast", "entropy"]]
# y = data[["output"]]
# X_train , X_test , y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.20)

