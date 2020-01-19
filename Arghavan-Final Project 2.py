#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pylab as plt


# In[2]:


import matplotlib
import matplotlib.pyplot as plt


# In[3]:


import seaborn as sb


# In[4]:


df = pd.read_csv('data.csv',error_bad_lines=False)


# In[5]:


df=df.rename(columns={'Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22\tQ23\tQ24\tQ25\tQ26\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tE13\tE14\tE15\tE16\tE17\tE18\tE19\tE20\tE21\tE22\tE23\tE24\tE25\tE26\tNIS_score\tcountry\tintroelapse\ttestelapse\tsurveyelapse\tTIPI1\tTIPI2\tTIPI3\tTIPI4\tTIPI5\tTIPI6\tTIPI7\tTIPI8\tTIPI9\tTIPI10\tVCL1\tVCL2\tVCL3\tVCL4\tVCL5\tVCL6\tVCL7\tVCL8\tVCL9\tVCL10\tVCL11\tVCL12\tVCL13\tVCL14\tVCL15\tVCL16\teducation\turban\tgender\tengnat\tage\thand\treligion\torientation\trace\tvoted\tmarried\tfamilysize\tmajor\t\t':'Result'})


# In[6]:


df = df.replace(to_replace= r"\t", value=" ", regex=True)


# In[7]:


df2 = pd.DataFrame(df.Result.str.split(" " ,95).tolist(),columns = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E15','E16','E17','E18','E19','E20','E21','E22','E23','E24','E25','E26','NIS_score','country','introelapse','testelapse','surveyelapse','TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10','VCL1','VCL2','VCL3','VCL4','VCL5','VCL6','VCL7','VCL8','VCL9','VCL10','VCL11','VCL12','VCL13','VCL14','VCL15','VCL16','education','urban','gender','engnat','age','hand','religion','orientation','race','voted','married','familysize','major'])


# In[8]:


dfnan=df2.where(df2 != '0', np.nan)


# In[9]:


dftipi=dfnan[['TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10','NIS_score','gender']]


# In[10]:


cols = dftipi.columns
nn = len(cols)
for i in range(nn):
    dftipi[cols[i]]=dftipi[cols[i]].str.replace(',','').astype(float)


# In[63]:


dftipi.mean(axis = 0) 


# In[65]:


dftipi.std()


# In[56]:


s=dftipi.std()
s


# In[66]:


dfM=dftipi[dftipi['gender']==1]


# In[67]:


dfM.mean()


# In[59]:


dfM.std()


# In[61]:


dff=dftipi[dftipi['gender']==2]


# In[62]:


dff.mean()


# In[68]:


dff.std()


# In[11]:


dfo=dftipi[dftipi['gender']==3]


# In[ ]:





# In[12]:


dfo.mean()


# In[13]:


dfo.std()


# In[ ]:




