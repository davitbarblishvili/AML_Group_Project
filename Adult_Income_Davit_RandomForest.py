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


df["workclass"]=df["workclass"].replace("?",np.nan)
df["occupation"]=df["occupation"].replace("?",np.nan)
df["native.country"]=df["native.country"].replace("?",np.nan)


# In[7]:


plt.figure(figsize = (12, 6))
df.isna().sum().plot.bar();


# Let's see how much data we would lose if we just dropped the null values.

# In[8]:


df2 = df.dropna()
percent_dropped = (df.shape[0] - df2.shape[0])/df.shape[0]*100
print(percent_dropped)


# 7% is significant so let's replace the values with the mode of each column

# In[9]:


df["workclass"]=df["workclass"].fillna(df["workclass"].mode()[0])
df["occupation"]=df["occupation"].fillna(df["occupation"].mode()[0])
df["native.country"]=df["native.country"].fillna(df["native.country"].mode()[0])


# We should also check for duplicates and only keep one of each unique value.

# In[10]:


df.duplicated().sum()


# In[11]:


df=df.drop_duplicates(keep="first")


# # Data Analysis

# ## 2.1 Univariate Analysis

# In[12]:


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

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig=plt.figure(figsize=(10,6))
ax=sns.countplot(df["workclass"])
plt.title("Working class distribution")   

fig.show()


# In[14]:


# Creating a barplot for 'Income'
income = df['income'].value_counts()

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(7, 5))
sns.barplot(income.index, income.values, palette='bright')
plt.title('Distribution of Income', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Income', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()


# In[15]:


# Creating a distribution plot for 'Age'
age = df['age'].value_counts()

plt.figure(figsize=(10, 5))
plt.style.use('fivethirtyeight')
sns.distplot(df['age'], bins=20)
plt.title('Distribution of Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()


# In[16]:


# Creating a barplot for 'Education'
edu = df['education'].value_counts()

plt.style.use('seaborn')
plt.figure(figsize=(10, 5))
sns.barplot(edu.values, edu.index, palette='Paired')
plt.title('Distribution of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# In[17]:


# Creating a barplot for 'Years of Education'
edu_num = df['education.num'].value_counts()

plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
sns.barplot(edu_num.index, edu_num.values, palette='colorblind')
plt.title('Distribution of Years of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Years of Education', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# In[18]:


# Creating a pie chart for 'Marital status'
marital = df['marital.status'].value_counts()

plt.style.use('default')
plt.figure(figsize=(10, 7))
plt.pie(marital.values, labels=marital.index, startangle=10, explode=(
    0, 0.20, 0, 0, 0, 0, 0), shadow=True, autopct='%1.1f%%')
plt.title('Marital distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.legend()
plt.legend(prop={'size': 7})
plt.axis('equal')
plt.show()


# In[19]:


# Creating a donut chart for 'Age'
relation = df['relationship'].value_counts()

plt.style.use('bmh')
plt.figure(figsize=(20, 10))
plt.pie(relation.values, labels=relation.index,
        startangle=50, autopct='%1.1f%%')
centre_circle = plt.Circle((0, 0), 0.7, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Relationship distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 30, 'fontweight': 'bold'})
plt.axis('equal')
plt.legend(prop={'size': 15})
plt.show()


# In[20]:


# Creating a barplot for 'Sex'
sex = df['sex'].value_counts()

plt.style.use('default')
plt.figure(figsize=(7, 5))
sns.barplot(sex.index, sex.values)
plt.title('Distribution of Sex', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Sex', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.grid()
plt.show()


# In[21]:


# Creating a Treemap for 'Race'
import squarify
race = df['race'].value_counts()

plt.style.use('default')
plt.figure(figsize=(7, 5))
squarify.plot(sizes=race.values, label=race.index, value=race.values)
plt.title('Race distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.show()


# In[22]:


# Creating a barplot for 'Hours per week'
hours = df['hours.per.week'].value_counts().head(10)

plt.style.use('bmh')
plt.figure(figsize=(15, 7))
sns.barplot(hours.index, hours.values, palette='colorblind')
plt.title('Distribution of Hours of work per week', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Hours of work', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# ## 2.2 Bivariate Analysis

# In[23]:


# Creating a countplot of income across age
plt.style.use('default')
plt.figure(figsize=(20, 7))
sns.countplot(df['age'], hue=df['income'])
plt.title('Distribution of Income across Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[24]:


# Creating a countplot of income across education
plt.style.use('seaborn')
plt.figure(figsize=(20, 7))
sns.countplot(df['education'],
              hue=df['income'], palette='colorblind')
plt.title('Distribution of Income across Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[25]:


# Creating a countplot of income across years of education
plt.style.use('bmh')
plt.figure(figsize=(20, 7))
sns.countplot(df['education.num'],
              hue=df['income'])
plt.title('Income across Years of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Years of Education', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.savefig('bi2.png')
plt.show()


# In[26]:


# Creating a countplot of income across Marital Status
plt.style.use('seaborn')
plt.figure(figsize=(20, 7))
sns.countplot(df['marital.status'], hue=df['income'])
plt.title('Income across Marital Status', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Marital Status', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[27]:


# Creating a countplot of income across race
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 7))
sns.countplot(df['race'], hue=df['income'])
plt.title('Distribution of income across race', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Race', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[28]:


# Creating a countplot of income across sex
plt.style.use('fivethirtyeight')
plt.figure(figsize=(7, 3))
sns.countplot(df['sex'], hue=df['income'])
plt.title('Distribution of income across sex', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Sex', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 10})
plt.savefig('bi3.png')
plt.show()


# ## 2.3 Multivariate Analysis

# In[29]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[30]:


df['income'] = le.fit_transform(df['income'])


# In[31]:


# Creating a pairplot of dataset
sns.pairplot(df)
plt.savefig('multi1.png')
plt.show()


# In[32]:


corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True,
                     annot=True, cmap='RdYlGn')
plt.savefig('multi2.png')
plt.show()


# # Using train.csv and test.csv
# 

# In[34]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_dev = train_df.drop('income', axis=1)
y_dev = train_df['income']
X_test = test_df.drop('income', axis=1)
y_test = test_df['income']

scaler = MinMaxScaler()
X_dev = scaler.fit_transform(X_dev)
X_test = scaler.transform(X_test)


# # Test/Train Split

# In[35]:


print("X_dev shape:", X_dev.shape)
print("X_test shape:", X_test.shape)
print("y_dev shape:", y_dev.shape)
print("y_test shape:", y_test.shape)


# In[36]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)

ros.fit(X_dev, y_dev)

X_resampled, Y_resampled = ros.fit_resample(X_dev, y_dev)
round(Y_resampled.value_counts(normalize=True) * 100, 2).astype('str') + ' %'


# # Random Forest Classifier

# In[37]:


from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(random_state=42)


# In[38]:


ran_for.fit(X_resampled, Y_resampled)
Y_pred_ran_for = ran_for.predict(X_test)


# # RandomForest Evaluation

# In[39]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[40]:


print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(y_test, Y_pred_ran_for) * 100, 2))
print('F1 score:', round(f1_score(y_test, Y_pred_ran_for) * 100, 2))


# # Hyperparameter tuning
# 

# In[41]:


from sklearn.model_selection import RandomizedSearchCV


# In[42]:


n_estimators = [int(x) for x in np.linspace(start=40, stop=150, num=15)]
max_depth = [int(x) for x in np.linspace(40, 150, num=15)]


# In[43]:


param_dist = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
}


# In[44]:


rf_tuned = RandomForestClassifier(random_state=42)


# In[45]:


rf_cv = RandomizedSearchCV(
    estimator=rf_tuned, param_distributions=param_dist, cv=5, random_state=42)


# In[46]:


print(f"Training on X_dev with {X_dev.shape[0]} samples")
t_start = time.time()

rf_cv.fit(X_resampled, Y_resampled)

t_end = time.time()
print(f"Random Forest train time = {t_end - t_start}")


# In[47]:


rf_cv.best_score_


# In[48]:


rf_cv.best_params_


# In[49]:


rf_best = RandomForestClassifier(
    max_depth=102, n_estimators=40, random_state=42)


# In[50]:


rf_best.fit(X_resampled, Y_resampled)


# In[51]:


p_start = time.time()
Y_pred_rf_best = rf_best.predict(X_test)
p_end = time.time()
print(f"Random Forest prediction time = {p_end - p_start}")


# In[52]:


print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(y_test, Y_pred_rf_best) * 100, 2))
print('F1 score:', round(f1_score(y_test, Y_pred_rf_best) * 100, 2))


# In[53]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred_rf_best)


# In[54]:


plt.style.use('default')
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.savefig('heatmap.png')
plt.show()


# In[55]:


from sklearn.metrics import classification_report
print(classification_report(y_test, Y_pred_rf_best))

