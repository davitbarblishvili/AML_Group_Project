import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#get data 
df = pd.read_csv("adult.csv")

#remove redundant columns or unnecessary columns
df = df.drop(['education','fnlwgt'],axis=1)

#replace ? with nan for easier removal
df['workclass'] = df['workclass'].replace("?",np.nan)
df['occupation'] = df['occupation'].replace("?",np.nan)
df['native.country'] = df['native.country'].replace("?",np.nan)

#replace nan with mode
df['workclass']=df['workclass'].fillna(df['workclass'].mode()[0])
df['occupation']=df['occupation'].fillna(df['occupation'].mode()[0])
df['native.country']=df['native.country'].fillna(df['native.country'].mode()[0])

#remove duplicates
df = df.drop_duplicates(keep='first')
#target variable coding
df['income'].replace(['<=50K','>50K'],[0,1], inplace=True)




#group and set native.country (domestic/international encoding)
df['native.country']=np.where(df['native.country'] =='United-States', 1, 0)
#turn sex into binary for male
df['sex']=np.where(df['sex'] =='Male', 1, 0)

#rename above 2 columns accordingly
df = df.rename(columns={"native.country": "native", "sex": "male"})

#categorical into one-hot-encoding (adapted from https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)
categorical_columns =['workclass','marital.status','occupation','relationship','race']
for col in categorical_columns:
    cat_list = pd.get_dummies(df[col], prefix=col)
    df=df.join(cat_list)

#remove old columns
df = df.drop(columns=categorical_columns)



#remove redundant columns (based on meta data)
df = df.drop(columns=['relationship_Husband','relationship_Unmarried', 'relationship_Wife'])

#statified split into test/train
train1, test1 = train_test_split(df,stratify=df['income'],test_size=0.25,random_state=42)
train2, test2 = train_test_split(df,stratify=df['income'],test_size=0.25,random_state=43)
train3, test3 = train_test_split(df,stratify=df['income'],test_size=0.25,random_state=44)

#write to file
train1.to_csv('train1.csv', index=False)
test1.to_csv('test1.csv', index=False)
train2.to_csv('train2.csv', index=False)
test2.to_csv('test2.csv', index=False)
train3.to_csv('train3.csv', index=False)
test3.to_csv('test3.csv', index=False)
