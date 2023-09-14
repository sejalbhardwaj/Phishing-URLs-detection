#!/usr/bin/env python
# coding: utf-8

# # Phishing Website Detection 

# Step -1 : Data preprocessing 
# 
# This dataset contains few website links (Some of them are legitimate websites and a few are fake websites)
# 
# Pre-Processing the data before building a model and also Extracting the features from the data based on certain conditions

# In[61]:


#importing numpy and pandas which are required for data pre-processing
import numpy as np
import pandas as pd


# In[62]:


#Loading the data
raw_data = pd.read_csv("traindataset.csv") 


# In[63]:


raw_data.head()


# We need to split the data according to parts of the URL
# 
# A typical URL could have the form http://www.example.com/index.html, which indicates a protocol (http), a hostname (www.example.com), and a file name (index.html).

# In[64]:


raw_data['URL'].str.split("://").head() #Here we divided the protocol from the entire URL. but need it to be divided it 
                                                 #seperate column


# In[65]:


seperation_of_protocol = raw_data['URL'].str.split("://",expand = True) #expand argument in the split method will give you a new column


# In[66]:


seperation_of_protocol.head()


# In[67]:


type(seperation_of_protocol)


# In[68]:


seperation_domain_name = seperation_of_protocol[1].str.split("/",1,expand = True) #split(seperator,no of splits according to seperator(delimiter),expand)


# In[69]:


type(seperation_domain_name)


# In[70]:


seperation_domain_name.columns=["domain_name","address"] #renaming columns of data frame


# In[71]:


seperation_domain_name.head()


# In[72]:


#Concatenation of data frames
splitted_data = pd.concat([seperation_of_protocol[0],seperation_domain_name],axis=1)


# In[73]:


splitted_data.columns = ['protocol','domain_name','address']


# In[74]:


splitted_data.head()


# In[75]:


splitted_data['is_phished'] = pd.Series(raw_data['Target'], index=splitted_data.index)


# In[76]:


splitted_data


# Domain name column can be further sub divided into domain_names as well as sub_domain_names 
# 
# Similarly, address column can also be further sub divided into path,query_string,file..................

# In[77]:


type(splitted_data)


# ### Features Extraction

# 
# Feature-1
# 
# 1.Long URL to Hide the Suspicious Part
# 
# If the length of the URL is greater than or equal 54 characters then the URL classified as phishing
# 
# 
# 0 --- indicates legitimate
# 
# 1 --- indicates Phishing
# 
# 2 --- indicates Suspicious

# In[78]:


def long_url(l):
    l= str(l)
    """This function is defined in order to differntiate website based on the length of the URL"""
    if len(l) < 54:
        return 0
    elif len(l) >= 54 and len(l) <= 75:
        return 2
    return 1


# In[79]:


#Applying the above defined function in order to divide the websites into 3 categories
splitted_data['long_url'] = raw_data['URL'].apply(long_url) 


# In[80]:


#Will show the results only the websites which are legitimate according to above condition as 0 is legitimate website
splitted_data[splitted_data.long_url == 0] 


# Feature-2
# 
# 2.URL’s having “@” Symbol
# 
# Using “@” symbol in the URL leads the browser to ignore everything preceding the “@” symbol and the real address often follows the “@” symbol.
# 
# IF {Url Having @ Symbol→ Phishing
#     Otherwise→ Legitimate }
# 
# 
# 0 --- indicates legitimate
# 
# 1 --- indicates Phishing
# 

# In[81]:


def have_at_symbol(l):
    """This function is used to check whether the URL contains @ symbol or not"""
    if "@" in str(l):
        return 1
    return 0
    


# In[82]:


splitted_data['having_@_symbol'] = raw_data['URL'].apply(have_at_symbol)


# In[83]:


splitted_data


# Feature-3
# 
# 3.Redirecting using “//”
# 
# The existence of “//” within the URL path means that the user will be redirected to another website.
# An example of such URL’s is: “http://www.legitimate.com//http://www.phishing.com”. 
# We examine the location where the “//” appears. 
# We find that if the URL starts with “HTTP”, that means the “//” should appear in the sixth position. 
# However, if the URL employs “HTTPS” then the “//” should appear in seventh position.
# 
# IF {ThePosition of the Last Occurrence of "//" in the URL > 7→ Phishing
#     
#     Otherwise→ Legitimate
# 
# 0 --- indicates legitimate
# 
# 1 --- indicates Phishing
# 

# In[84]:


def redirection(l):
    """If the url has symbol(//) after protocol then such URL is to be classified as phishing """
    if "//" in str(l):
        return 1
    return 0


# In[85]:


splitted_data['redirection_//_symbol'] = seperation_of_protocol[1].apply(redirection)


# In[86]:


splitted_data.head()


# Feature-4
# 
# 4.Adding Prefix or Suffix Separated by (-) to the Domain
# 
# The dash symbol is rarely used in legitimate URLs. Phishers tend to add prefixes or suffixes separated by (-) to the domain name
# so that users feel that they are dealing with a legitimate webpage. 
# 
# For example http://www.Confirme-paypal.com/.
#     
# IF {Domain Name Part Includes (−) Symbol → Phishing
#     
#     Otherwise → Legitimate
#     
# 1 --> indicates phishing
# 
# 0 --> indicates legitimate
#     

# In[87]:


def prefix_suffix_seperation(l):
    if '-' in str(l):
        return 1
    return 0


# In[88]:


splitted_data['prefix_suffix_seperation'] = seperation_domain_name['domain_name'].apply(prefix_suffix_seperation)


# In[89]:


splitted_data.head()


# Feature - 5
# 
# 5. Sub-Domain and Multi Sub-Domains
# 
# The legitimate URL link has two dots in the URL since we can ignore typing “www.”. 
# If the number of dots is equal to three then the URL is classified as “Suspicious” since it has one sub-domain.
# However, if the dots are greater than three it is classified as “Phishy” since it will have multiple sub-domains
# 
# 0 --- indicates legitimate
# 
# 1 --- indicates Phishing
# 
# 2 --- indicates Suspicious
# 

# In[90]:


def sub_domains(l):
    l= str(l)
    if l.count('.') < 3:
        return 0
    elif l.count('.') == 3:
        return 2
    return 1


# In[91]:


splitted_data['sub_domains'] = splitted_data['domain_name'].apply(sub_domains)


# In[92]:


splitted_data


# ### Classification of URLs using Random forest 

# In[93]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# In[94]:


#Features
x = splitted_data.columns[4:9]
x   


# In[95]:


#variable to be predicted; yes = 0 and no = 1
y = pd.factorize(splitted_data['is_phished'])[0]
y 


# In[96]:


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators=100,n_jobs=2,random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(splitted_data[x], y)


# In[97]:


test_data = pd.read_csv("testdataset.csv") 


# In[98]:


clf.predict(test_data[x]) #testing the classifier on test data.


# In[99]:


clf.predict_proba(test_data[x])[0:10] #predicted probability for each class.


# ### Evaluating classifier

# In[100]:


preds = test_data.is_phished[clf.predict(test_data[x])] #predicted values


# In[101]:


preds.head(10)


# In[102]:


actual = pd.Series(test_data['is_phished']) #actual values


# In[103]:


confusion_matrix(actual,preds) 


# In[104]:


accuracy_score(actual,preds) #accuracy of classifier


# In[105]:


#importance of features
list(zip(splitted_data[x], clf.feature_importances_))


# In[107]:


# Load the pre-trained model
import pickle
filename = 'finalmodel.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to extract features from a URL
def extract_features(url):
    features = {
        'long_url': len(url),
        'having_@_symbol': 1 if "@" in url else 0,
        'redirection_//_symbol': 1 if "//" in url else 0,
        'prefix_suffix_seperation': 1 if "-" in url else 0,
        'sub_domains': url.count('.')
    }
    return pd.DataFrame([features])

# Get URL input from the user
user_url = input("Enter a URL: ")

# Extract features from the user-provided URL
new_data = extract_features(user_url)

# Make predictions
prediction = loaded_model.predict(new_data)

# Interpret the prediction (0 for legitimate, 1 for phishing)
if prediction == 0:
    result = "Legitimate"
else:
    result = "Phishing"

print(f"The URL is classified as: {result}")


# In[ ]:




