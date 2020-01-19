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


df.shape


# In[6]:


df.columns


# In[7]:


df.head()


# In[8]:


df=df.rename(columns={'Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22\tQ23\tQ24\tQ25\tQ26\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tE13\tE14\tE15\tE16\tE17\tE18\tE19\tE20\tE21\tE22\tE23\tE24\tE25\tE26\tNIS_score\tcountry\tintroelapse\ttestelapse\tsurveyelapse\tTIPI1\tTIPI2\tTIPI3\tTIPI4\tTIPI5\tTIPI6\tTIPI7\tTIPI8\tTIPI9\tTIPI10\tVCL1\tVCL2\tVCL3\tVCL4\tVCL5\tVCL6\tVCL7\tVCL8\tVCL9\tVCL10\tVCL11\tVCL12\tVCL13\tVCL14\tVCL15\tVCL16\teducation\turban\tgender\tengnat\tage\thand\treligion\torientation\trace\tvoted\tmarried\tfamilysize\tmajor\t\t':'Result'})


# In[9]:


df.head()


# In[10]:


df = df.replace(to_replace= r"\t", value=" ", regex=True)


# In[327]:


df.head(5)


# In[11]:


df2 = pd.DataFrame(df.Result.str.split(" " ,95).tolist(),columns = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','E15','E16','E17','E18','E19','E20','E21','E22','E23','E24','E25','E26','NIS_score','country','introelapse','testelapse','surveyelapse','TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10','VCL1','VCL2','VCL3','VCL4','VCL5','VCL6','VCL7','VCL8','VCL9','VCL10','VCL11','VCL12','VCL13','VCL14','VCL15','VCL16','education','urban','gender','engnat','age','hand','religion','orientation','race','voted','married','familysize','major'])


# In[12]:


df2.head(5)


# In[13]:


df2.shape


# ## Gender

# In[14]:


df2['gender'].value_counts()


# In[15]:


df2['gender'].value_counts(normalize=True)


# In[394]:



# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Female', 'Male', 'Other', 'No Answer'
sizes = [74.1, 12.9, 12.7, 0.3]
explode = (0, 0, 0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


ax1.set_title("Gender")

plt.show()
plt.savefig('Gender.png')
#files.download("Gender.png")


# # hand

# In[389]:


df2['hand'].value_counts()


# In[390]:


df2['hand'].value_counts(normalize=True)


# In[391]:


labels = 'Right', 'Left', 'Both', 'No Answer'
sizes = [86.9, 9.7, 3.0, 0.4]
explode = (0, 0, 0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


ax1.set_title("Hand")

plt.show()


# # Education

# In[485]:


df2['education'].value_counts(normalize=True)


# In[80]:


labels = 'Less than high school','High school', 'University degree', 'Graduate degree','No Answer'
sizes = [51.7, 24.3, 18.4, 4.7,0.9]
explode = (0, 0, 0,0,0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


ax1.set_title("Education")

plt.show()


# ## Races

# In[490]:


df2['race'].value_counts(normalize=True)


# In[ ]:


labels = 10=Asian, 20=Arab, 30=Black, 40=Indigenous Australian, 50=Native American, 60=White, 70=Other
sizes = ['0']
explode = (0, 0, 0, 0)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


ax1.set_title("Hand")

plt.show()


# ## Urban

# ## Age

# In[259]:


df2['age'].max()


# In[260]:


df2[df2['age']==df2['age'].max()]


# In[16]:


dfae=df2.loc[df2['age']=='0']


# In[262]:


dfae.shape


# In[17]:


def age_group(age):
    
   
    age = int(age)
    
    if age <1:
        bucket = 'No Answer'
    
    if age in range (1,20):
        bucket = '<20'
    
   
    if age in range(20, 30):
        bucket = '20-29'
        
    if age in range(30, 40):
        bucket = '30-39'
        
    if age in range(40, 50):
        bucket = '40-49'
    
    if age in range(50, 60):
        bucket = '50-59'
        
    if age in range(60, 70):
        bucket = '60-69'
    
    if age >= 70:
        bucket = '70+'

    return bucket 


# In[18]:


df2['age_group'] = df2['age'].apply(age_group)


# In[265]:


df2['age_group'].value_counts()


# In[266]:


df2['age_group'].value_counts().plot(kind="bar", figsize=(15,7), color="#61d199")
plt.title('Age Distribution')
plt.xlabel('Age group')
plt.ylabel('Number')
 


# ## Language

# In[22]:


df2['engnat'].value_counts()


# In[23]:


df2['engnat'].value_counts(normalize=True)


# In[483]:


labels = 'English', 'other', 'No Answer'
sizes = [77.5, 22.3, 0.2]
explode = (0, 0, 0, )  

fig3, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


ax1.set_title("Language")

plt.show()


# ## A value of 1 is checked, 0 means unchecked. The words at VCL6, VCL9, and VCL12 are not real words and can be used as a validity check.
# 

# In[33]:


dfvcl2=df2[df2['VCL9']==1]


# In[34]:


dfvcl2


# # Correlation

# ## 1.General corr

# In[19]:


#dfcn=df2.replace("0","nan")

dfcn=df2.where(df2 != '0', np.nan)


# In[317]:


#dfcn


# In[20]:


dfgc=dfcn.drop(['major','country','age_group','VCL1','VCL2','VCL3','VCL4','VCL5','VCL6','VCL7','VCL8','VCL9','VCL10','VCL11','VCL12','VCL13','VCL14','VCL15','VCL16'], axis=1)


# In[457]:


dfgc.shape


# In[21]:


cols = dfgc.columns
nn = len(cols)
for i in range(nn):
    dfgc[cols[i]]=dfgc[cols[i]].str.replace(',','').astype(float)


GC=dfgc.corr(method ='pearson') 


# In[22]:


plt.matshow(GC)
plt.show()


# In[23]:



GC.to_csv('GC.csv')


# In[68]:


a = GC.unstack()
aa = a.sort_values(kind="quicksort")


# In[433]:


plt.figure(figsize=(100,80))
sb.heatmap(GC, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[69]:


aa.to_csv('aa.csv')


# In[ ]:





# In[75]:


GCdata=GCdata.rename(columns={'-0.7773475435739544':'ok'}) #I Know!


# In[77]:


GCdata.head()


# In[78]:


GCresult=GCdata.loc[(GCdata['ok'] >= 0.7) & (GCdata['ok'] <= 0.95)]
GCresult
GCresult.to_csv('GCresult.csv')


# In[26]:


dfgc=dfcn.drop(['major','country','age_group','VCL1','VCL2','VCL3','VCL4','VCL5','VCL6','VCL7','VCL8','VCL9','VCL10','VCL11','VCL12','VCL13','VCL14','VCL15','VCL16'], axis=1)


# In[ ]:





# ### Gender & Q Questions

# In[27]:


dfgq=dfcn[['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','gender','NIS_score']]


# In[463]:


dfgq.head()


# In[28]:


cols = dfgq.columns
nn = len(cols)
for i in range(nn):
    dfgq[cols[i]]=dfgq[cols[i]].str.replace(',','').astype(float)


gqc=dfgq.corr(method='pearson')


# In[470]:


plt.figure(figsize=(20,15))
sb.heatmap(gqc, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[73]:


m= gqc.unstack()
mm =m.sort_values(kind="quicksort")





# In[32]:


mm.to_csv('mm.csv')


# In[45]:


cdata = pd.read_csv('mm.csv',error_bad_lines=False)
cdata.head()
#I know the result is not clean


# In[48]:


cdata.columns


# In[47]:


cdata=cdata.rename(columns={'-0.7773475435739544':'ok'}) #I Know!


# In[57]:


Qcresult=cdata.loc[(cdata['ok'] >= 0.7) & (cdata['ok'] <= 0.95)]
Qcresult
Qcresult.to_csv('Qcresult.csv')


# In[58]:


Qcresultmanfi=cdata.loc[(cdata['ok'] >= -0.95) & (cdata['ok'] <= -0.7)]
Qcresultmanfi
Qcresultmanfi.to_csv('Qcresultmanfi.csv')


# In[481]:


dfgq['Qsum']=dfgq['Q1']+dfgq['Q2']+dfgq['Q3']+dfgq['Q4']+dfgq['Q5']+dfgq['Q6']+dfgq['Q7']+dfgq['Q8']+dfgq['Q9']+dfgq['Q10']+dfgq['Q11']+dfgq['Q12']+dfgq['Q13']+dfgq['Q14']+dfgq['Q15']+dfgq['Q16']+dfgq['Q17']+dfgq['Q18']+dfgq['Q19']+dfgq['Q20']+dfgq['Q21']+dfgq['Q22']+dfgq['Q23']+dfgq['Q24']+dfgq['Q25']+dfgq['Q26']
dfgq.head(5)


# In[472]:


dfgqM=dfgq[dfgq['gender']==1]
dfgqF=dfgq[dfgq['gender']==2]
dfgqO=dfgq[dfgq['gender']==3]


# In[473]:


QMA=dfgqM['Qsum'].mean()
QFA=dfgqF['Qsum'].mean()
QOA=dfgqO['Qsum'].mean()


# In[474]:


QMA


# In[475]:


QFA


# In[476]:


QOA
# there is realy no correlaion 
#  the minimun possible sum is 26 and  the maximum is 130


# In[478]:


dftest=dfgq[['NIS_score','Qsum']]
#dftest


# ### Gender & TIPI &hand

# In[398]:


dfgth=dfcn[['TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10','gender','hand']]


# In[399]:


cols = dfgth.columns
nn = len(cols)
for i in range(nn):
    dfgth[cols[i]]=dfgth[cols[i]].str.replace(',','').astype(float)


gth=dfgth.corr(method='pearson')


# In[480]:


dfgth['Tsum']=dfgth['TIPI1']+dfgth['TIPI2']+dfgth['TIPI3']+dfgth['TIPI4']+dfgth['TIPI5']+dfgth['TIPI6']+dfgth['TIPI7']+dfgth['TIPI8']+dfgth['TIPI9']+dfgth['TIPI10']
dfgth.head(5)


# In[400]:


plt.figure(figsize=(20,15))
sb.heatmap(gth, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[59]:


dfmohem=dfcn[['TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10','education','urban','gender','engnat','age','hand','religion','orientation','race','voted','married','familysize']]


# In[61]:


cols =dfmohem.columns
nn = len(cols)
for i in range(nn):
    dfmohem[cols[i]]=dfmohem[cols[i]].str.replace(',','').astype(float)


mohem=dfmohem.corr(method='pearson')


# In[62]:


plt.figure(figsize=(20,15))
sb.heatmap(mohem, annot=True, cmap=plt.cm.Reds)
plt.show()


# ## 3####

# In[ ]:





# In[302]:


dfQTIPI=df2[['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q24','Q25','Q26','TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10']]


# In[274]:


dfQTIPI.shape


# In[ ]:





# In[81]:


cols = dfQTIPI.columns
nn = len(cols)
for i in range(nn):
    dfQTIPI[cols[i]]=dfQTIPI[cols[i]].str.replace(',','').astype(float)


correlation=dfQTIPI.corr(method ='pearson') 


# In[314]:


plt.matshow(correlation)
plt.show()


# In[305]:


#correlation


# In[313]:


s = correlation.unstack()
so = s.sort_values(kind="quicksort")


# In[ ]:





# In[280]:


import seaborn as sb


# In[281]:


plt.figure(figsize=(20,15))
sb.heatmap(correlation, 
            xticklabels=correlation.columns,
            yticklabels=correlation.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)


# In[ ]:




