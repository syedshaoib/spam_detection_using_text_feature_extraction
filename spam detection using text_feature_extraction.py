#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df = pd.read_csv('../TextFiles/smsspamcollection.tsv', sep='\t')
df.head()


# In[4]:


x=df['message']
y=df['label']


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)


# In[10]:


# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# x_train_vocab=count_vect.fit_transform(x_train)


# In[11]:


x_train_vocab.shape


# In[12]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[20]:


tfidf=TfidfTransformer()
x_train_tfidf=tfidf.fit_transform(x_train_vocab)


# In[18]:


from sklearn.svm import LinearSVC
clf = LinearSVC()

clf.fit(x_train_tfidf,y_train)


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()


# In[24]:


from sklearn.pipeline import Pipeline

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])


# In[ ]:





# In[ ]:


# from sklearn.pipeline import Pipeline
# pipline=Pipline([('')])


# In[ ]:





# In[29]:


pred=text_clf.predict(x_test)


# In[31]:


from sklearn import metrics


# In[32]:


metrics.confusion_matrix(y_test,pred)


# In[34]:


metrics.accuracy_score(y_test,pred)


# In[42]:


type(x_train)


# In[ ]:




