#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


# In[2]:


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# # Data Cleaning

# In[3]:


df=pd.read_csv("adult.csv")
df


# In[4]:


df.dtypes


# In[5]:


df.isna().sum()


# This tells us the dataset has no missing values, but upon inspection, we see several columns having "?" as values.

# In[6]:


df.describe(include='all')


# In[7]:


df["workclass"]=df["workclass"].replace("?",np.nan)
df["occupation"]=df["occupation"].replace("?",np.nan)
df["native.country"]=df["native.country"].replace("?",np.nan)


# In[8]:


plt.figure(figsize = (12, 6))
df.isna().sum().plot.bar();


# Let's see how much data we would lose if we just dropped the null values.

# In[9]:


df2 = df.dropna()
percent_dropped = (df.shape[0] - df2.shape[0])/df.shape[0]*100
print(percent_dropped)


# 7% is significant so let's replace the values with the mode of each column

# In[10]:


df["workclass"]=df["workclass"].fillna(df["workclass"].mode()[0])
df["occupation"]=df["occupation"].fillna(df["occupation"].mode()[0])
df["native.country"]=df["native.country"].fillna(df["native.country"].mode()[0])


# We should also check for duplicates and only keep one of each unique value.

# In[11]:


df.duplicated().sum()


# In[12]:


df=df.drop_duplicates(keep="first")


# # Data Analysis

# In[13]:


fig =  plt.figure(figsize = (15,6))
fig.patch.set_facecolor('#bcd9e6')
                                                 
gs = fig.add_gridspec(2,3)
gs.update(wspace=0.2,hspace= 0.2)

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[1,2])

axes=[ax0,ax1,ax2,ax3,ax4,ax5]
for ax in axes:
    ax.set_facecolor('#f5f6f6')
    ax.tick_params(axis='x',
                   labelsize = 12, which = 'major',
                   direction = 'out',pad = 2,
                   length = 1.5)
    ax.tick_params(axis='y', colors= 'black')
    ax.axes.get_yaxis().set_visible(False)
    
    for loc in ['left', 'right', 'top', 'bottom']:
        ax.spines[loc].set_visible(False)

cols = df.select_dtypes(exclude = 'object').columns

sns.kdeplot(x = df[cols[0]],color="yellow",fill=True,ax = ax0)
sns.kdeplot(x = df[cols[1]],color="red",fill=True,ax = ax1)
sns.kdeplot(x = df[cols[2]],color="blue",fill=True,ax = ax2)
sns.kdeplot(x = df[cols[3]],color="black",fill=True,ax = ax3)
sns.kdeplot(x = df[cols[4]],color="pink",fill=True,ax = ax4)
sns.kdeplot(x = df[cols[5]],color="green",fill=True,ax = ax5)

fig.text(0.2,0.98,"Univariate Analysis on Numerical Columns:",**{'font':'serif', 'size':18,'weight':'bold'}, alpha = 1)


# From this we can see that the majority of adults are within the ages of 20-45 and most adults spend 40 hours a week working.

# In[14]:


income=df["income"].reset_index()
px.pie(values=income["index"],names=income["income"], color_discrete_sequence=px.colors.sequential.Hot, title='Income of Adults')


# In[15]:


sex=df["sex"].reset_index()
px.pie(values=sex["index"],names=sex["sex"],title='Sex Distribution', hole=.2)


# In[16]:


race=df["race"].reset_index()
px.pie(values=race["index"],names=race["race"], title='Race of Adults')


# In[17]:


relationship=df["relationship"].reset_index()
px.pie(values=relationship["index"],names=relationship["relationship"], title='Relationship Status')


# In[18]:


occupation=df["occupation"].reset_index()
px.pie(values=occupation["index"],names=occupation["occupation"], title='Occupation')


# In[19]:


marital_status=df["marital.status"].reset_index()
px.pie(values=marital_status["index"],names=marital_status["marital.status"], title='Marital Status')


# In[20]:


education=df["education"].reset_index()
px.pie(values=education["index"],names=education["education"], title='Education')


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig=plt.figure(figsize=(10,6))
ax=sns.countplot(df["workclass"])
plt.title("Working class distribution")   

fig.show()


# ## Using test.csv and train.csv
# 

# In[22]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[23]:


main_df = pd.concat([train_df, test_df])


# In[24]:


X_dev = main_df.drop('income', axis=1)
y_dev = main_df['income']


scaler = MinMaxScaler()
X_dev = scaler.fit_transform(X_dev)


# # Data Sampling and Train Splitting
# 

# In[25]:


n_samples = 20000

Data_sample = pd.DataFrame(X_dev).sample(n=n_samples, random_state=8).values
target_sample = pd.DataFrame(y_dev).sample(n=n_samples, random_state=8).values

print(Data_sample.shape)
print(target_sample.shape)


# In[26]:


from sklearn.model_selection import train_test_split

Data_sample_train, Data_sample_test, target_sample_train, target_sample_test = train_test_split(Data_sample, target_sample, 
                                                    test_size = 0.3, random_state=999,
                                                    stratify = target_sample)

print(Data_sample_train.shape)
print(Data_sample_test.shape)


# # model evaluation strategy
# 

# In[27]:


from sklearn.model_selection import StratifiedKFold, GridSearchCV

cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)


# In[28]:


from sklearn.base import BaseEstimator, TransformerMixin


class RFIFeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_features_=10):
        self.n_features_ = n_features_
        self.fs_indices_ = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from numpy import argsort
        model_rfi = RandomForestClassifier(n_estimators=100)
        model_rfi.fit(X, y)
        self.fs_indices_ = argsort(model_rfi.feature_importances_)[::-1][0:self.n_features_] 
        return self 

    def transform(self, X, y=None):
        return X[:, self.fs_indices_]


# In[29]:


from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

pipe_KNN = Pipeline(steps=[('rfi_fs', RFIFeatureSelector()), 
                           ('knn', KNeighborsClassifier())])

params_pipe_KNN = {'rfi_fs__n_features_': [10, 20, X_dev.shape[1]],
                   'knn__n_neighbors': [1, 5, 10, 15, 20],
                   'knn__p': [1, 2]}

gs_pipe_KNN = GridSearchCV(estimator=pipe_KNN, 
                           param_grid=params_pipe_KNN, 
                           cv=cv_method,
                           refit=True,
                           n_jobs=-2,
                           scoring='roc_auc',
                           verbose=1) 


# In[31]:


print(f"Training on X_dev with {Data_sample_train.shape[0]} samples")
t_start = time.time()

gs_pipe_KNN.fit(Data_sample_train, target_sample_train.ravel());

t_end = time.time()
print(f"KNN train time = {t_end - t_start}")


# In[32]:


gs_pipe_KNN.best_params_


# In[33]:


gs_pipe_KNN.best_score_


# In[34]:


def get_search_results(gs):

    def model_result(scores, params):
        scores = {'mean_score': np.mean(scores),
             'std_score': np.std(scores),
             'min_score': np.min(scores),
             'max_score': np.max(scores)}
        return pd.Series({**params,**scores})

    models = []
    scores = []

    for i in range(gs.n_splits_):
        key = f"split{i}_test_score"
        r = gs.cv_results_[key]        
        scores.append(r.reshape(-1,1))

    all_scores = np.hstack(scores)
    for p, s in zip(gs.cv_results_['params'], all_scores):
        models.append((model_result(s, p)))

    pipe_results = pd.concat(models, axis=1).T.sort_values(['mean_score'], ascending=False)

    columns_first = ['mean_score', 'std_score', 'max_score', 'min_score']
    columns = columns_first + [c for c in pipe_results.columns if c not in columns_first]

    return pipe_results[columns]


# In[35]:


results_KNN = get_search_results(gs_pipe_KNN)
results_KNN.head()


# # Evaluation and Performance

# In[36]:


results_KNN_10_features = results_KNN[results_KNN['rfi_fs__n_features_'] == 10.0]

for i in results_KNN_10_features['knn__p'].unique():
    temp = results_KNN_10_features[results_KNN_10_features['knn__p'] == i]
    plt.plot(temp['knn__n_neighbors'], temp['mean_score'], marker = '.', label = i)
    
plt.legend(title = "p")
plt.xlabel('Number of Neighbors')
plt.ylabel("AUC Score")
plt.title("KNN Performance Comparison with 10 Features")
plt.show()


# In[37]:


from sklearn.model_selection import cross_val_score

cv_method_ttest = StratifiedKFold(n_splits=10, shuffle=True, random_state=111)

cv_results_KNN = cross_val_score(estimator=gs_pipe_KNN.best_estimator_,
                                 X=Data_sample_test,
                                 y=target_sample_test.ravel(), 
                                 cv=cv_method_ttest, 
                                 n_jobs=-2,
                                 scoring='roc_auc')
cv_results_KNN.mean()


# In[39]:


p_start = time.time()
pred_KNN = gs_pipe_KNN.predict(Data_sample_test)
p_end = time.time()
print(f"KNN prediction time = {p_end - p_start}")


# # Statistics

# In[40]:


from sklearn import metrics
print("\nClassification report for K-Nearest Neighbor") 
print(metrics.classification_report(target_sample_test, pred_KNN))


# # Confusion Matrix

# In[41]:


from sklearn import metrics
print("\nConfusion matrix for K-Nearest Neighbor") 
print(metrics.confusion_matrix(target_sample_test, pred_KNN))


# In[42]:


from sklearn import metrics
print("\nConfusion matrix for K-Nearest Neighbor") 
cm = metrics.confusion_matrix(target_sample_test, pred_KNN)
cm_df = pd.DataFrame(cm, index = ['>50K','<=50K'],
columns = ['>50K','<=50K'])

# plot the confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("KNN confusion matrix")

