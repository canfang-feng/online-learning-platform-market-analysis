#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,ImageColorGenerator
from sklearn.preprocessing import StandardScaler


# In[2]:


sns.set()
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"]=(10,5)


# In[3]:


#### import the data sets
df=pd.read_csv("2016-FCC-New-Coders-Survey-Data - Copy - Copy.csv",low_memory=False)
df.head()


# In[4]:


# group the ages and visulize the distribution
df["AgeG"]=pd.cut(df.Age,[0,18,24,30,40,60,90],labels=["10-18","18-24","24-30","30-40","40-60","60-90"],right=False)
ax=df.AgeG.value_counts(sort=False).plot(kind="bar",rot=0,color="g")
ax.set_title("Age Distribution")
ax.set_xlabel("Age Groups")
ax.set_ylabel("numbers of respondents")


# In[5]:


# classify other genders and visulize the distribution
df["GenG"]=df.Gender.copy()
df.loc[df.GenG.isin(["genderqueer","trans","agender"])|df.GenG.isnull(),"GenG"]="other/null"

ax=df.GenG.value_counts(sort=True).plot(kind="bar",rot=0,color="g")
ax.set_title("Gender distribution")
ax.set_xlabel("Gender")
ax.set_ylabel("numbers of respondents")


# In[6]:


# analyze and visulize the relationship between income,age and gender
ax=sns.boxplot(data=df,x="AgeG",y="Income",hue="GenG", palette="Set3",fliersize=1,linewidth=1)
ax.set_title("Current income & Age Group & Gender")
ax.set_xlabel("Age Group")
ax.set_ylabel("Current income in dollars")
ax.set_yticks(np.linspace(0,200000,11))


# In[7]:


# analyze and visulize the relationship between school degree and income
ax=sns.boxplot(data=df,y="SchoolDegree",x="Income", palette="Set3",fliersize=2,linewidth=1)
ax.set_title("income & school degrees")
ax.set_xlabel("income (dollars)")
ax.set_ylabel("school degrees")
ax.set_xticks(np.linspace(0,200000,11))


# In[8]:


# analyze and visulize the relationship between different coding event.
event=df[df["CodeEventNone"]!=1][["CodeEventBootcamp","CodeEventCoffee","CodeEventConferences","CodeEventDjangoGirls","CodeEventGameJam",
             "CodeEventGirlDev","CodeEventHackathons","CodeEventMeetup","CodeEventNodeSchool",
              "CodeEventRailsBridge","CodeEventRailsGirls","CodeEventStartUpWknd","CodeEventWomenCode","CodeEventWorkshop"]].copy()

event=event.replace(np.nan,0)

corr=event.corr(method='pearson', min_periods=1)
ax=sns.heatmap(corr,vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
ax.set_title("The correlation between different coding events")


# In[9]:


# analyze and visulize the relationship between different learning resources
res=df[["ResourceBlogs","ResourceBooks","ResourceCodecademy","ResourceCodeWars",
             "ResourceCoursera","ResourceDevTips","ResourceEdX","ResourceEggHead","ResourceFCC",
             "ResourceGoogle","ResourceHackerRank","ResourceKhanAcademy","ResourceLynda","ResourceMDN",
             "ResourceOdinProj","ResourcePluralSight","ResourceReddit","ResourceSkillCrush","ResourceSoloLearn",
             "ResourceStackOverflow","ResourceTreehouse","ResourceUdacity","ResourceUdemy","ResourceW3Schools","ResourceYouTube"]].copy()

res=res.replace(np.nan,0)

corr=res.corr(method='pearson', min_periods=1)
ax=sns.heatmap(corr,vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
ax.set_title("The correlation between different resources")


# In[8]:


# create network to visulize the customers citizens and moving between different countries
import networkx as nx
net1=pd.read_csv("net1.csv")
net1=net1.fillna(0)
net=np.array(net1)

D=nx.DiGraph(net)
nodew=[]
nodew=[67,210,53,263,452,46,47,59,47,113,176,968,103,47,36,50,100,113,192,126,237,454,5756,51]
pos = nx.fruchterman_reingold_layout(D)
nx.draw_networkx(D,pos=pos,node_color="g",node_size=nodew,with_labels=True,linewidths=0.5)


# In[9]:


########## cluster the customer into different subgroups.

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df2=pd.read_csv("cluster1.csv")
df2=df2.fillna(0)


# In[10]:


### cluster_ transfer the string data into the numerical type 
name=[]
name=["CountryCitizen","CountryLive","EmploymentField","EmploymentStatus","JobApplyWhen","JobPref","JobRoleInterest","JobWherePref","LanguageAtHome","MaritalStatus","SchoolDegree","SchoolMajor"]

for i in name:
    df2[i]= pd.Categorical(df2[i])
    df2[i] = df2[i].cat.codes


# In[71]:


### cluster_ normalize the data
X=np.array(df2)
X = StandardScaler().fit_transform(X)
print(X.shape)


# In[97]:


### try different numbers of clustering 
data_num=X.shape[0]
err_clustering=np.zeros([21,1])

for k in range(21):
    k_means=KMeans(n_clusters=k+1, max_iter=300).fit(X)
    err_clustering[k]=k_means.inertia_/data_num

print(np.around(err_clustering, decimals=2, out=None))

fig=plt.figure()
plt.rcParams["figure.figsize"]=(10,5)
plt.plot(range(1,22),err_clustering)
plt.xlabel('Number of clusters')
plt.ylabel('Clustering error')
plt.title("The number of clusters vs clustering error")
ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
plt.show() 


# In[93]:


####### repeat k-means 50 times using k=10 clusters and L=100 interations in each repetition.

min_ind=0

X=np.zeros([15620,35])
X=np.array(df2)
X = StandardScaler().fit_transform(X)

cluster_assignment=np.zeros((50,X.shape[0]),dtype=np.int32)
clustering_err=np.zeros([50,1])

np.random.seed(42)
init_means_cluster1 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster2 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster3 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster4 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster5 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster6 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster7 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster8 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster9 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 
init_means_cluster10 = np.random.randn(50,35)  # use the rows of this numpy array to init k-means 


best_perform=np.zeros((15620,1))

init_means_cluster_ar=np.zeros((50,10,35))
for i in range(50):
    init_means_cluster_ar[i]=[init_means_cluster1[i,:],
                              init_means_cluster2[i,:],
                              init_means_cluster3[i,:],
                              init_means_cluster4[i,:],
                              init_means_cluster5[i,:],
                              init_means_cluster6[i,:],
                              init_means_cluster7[i,:],
                              init_means_cluster8[i,:],
                              init_means_cluster9[i,:],
                              init_means_cluster10[i,:]
                              ]
    
data_num = X.shape[0]

for i in range(50):
    k_means=KMeans(n_clusters = 10,init=init_means_cluster_ar[i],max_iter = 100).fit(X)
    err_clustering=k_means.inertia_/data_num
    clustering_err[i]=err_clustering
    cluster_assignment[i]=k_means.labels_

min_ind=np.argmin(clustering_err)
best_perform=cluster_assignment[min_ind]


# In[99]:


fig=plt.figure(figsize=(8,6))
plt.plot(range(1,51),clustering_err)
plt.xlabel('Number of repetition')
plt.ylabel('Clustering error')
plt.title("The number of repetition vs clustering error")
ax=plt.gca()
#ax.set_xticks([2,4,6,8,10,12,14,16,18,20])
plt.show() 


# In[109]:


# visulize the best clustering reuslt
plt.hist(best_perform)
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.show()


# In[107]:


# assign the cluster value in the dataset and create a new column, download the dataset for further analysis in tableau
df["cluster"]=best_perform
df.to_csv('result of cluster.csv')

