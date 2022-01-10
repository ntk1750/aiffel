#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


wins = load_wine()


# In[8]:


# 변수확인
print(wins.DESCR)


# In[3]:


wins.keys()


# 데이터 이해하기

# In[4]:


#Feature Data 지정하기
wins_data = wins.data
wins_data.shape


# In[5]:


#label data 지정하기
label = wins.target
label.shape


# In[6]:


#target name 출력하기
target_name = wins.target_names
target_name


# In[11]:


import pandas as pd
wins_df = pd.DataFrame(data=wins_data, columns=wins.feature_names)
wins_df


# In[12]:


wins_df["label"] = wins.target
wins_df


# In[13]:


# train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(wins_data,
                                                    label,
                                                    test_size=0.2,
                                                    random_state=15)


# In[14]:


# 데이터셋 확인 
X_train.shape, y_train.shape


# In[15]:


X_test.shape, y_test.shape


# #### Decision Tree 사용하기

# In[16]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)# 모델 저장
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred))


# #### Random Forest 사용하기

# In[17]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# #### SVM 사용하기

# In[18]:


from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# #### SGD Classifier 사용하기

# In[19]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# #### Logistic Regression 사용하기

# In[20]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))


# ### 평가지표

# load_wine 데이터를 사용하여 총 5가지의 알고리즘을 바탕으로 분석하였다.
# 첫번째 알고리즘인 Decision Tree의 경우 accuracy는 0.92, Random Forest accuracy 1 ,
# SVM accuracy 0.61, SGD Classifier accuracy 0.69, Logistic Regression accuracy 0.94로 다양한 값의 정확도가 나타났다.
# 
# 총 5가지 알고리즘 중 sklearn.metrics 의 평가지표를 바탕으로 wine 데이터에 적절한 
# 알고리즘으로는 Logistic Regression을 뽑았다.
# 
# Logistic Regression은 다른 알고리즘에 비해 accuracy이 2번째로 높은 정확도를 보였다.
# 가장 높은 정확도를 보이니 Random Forest 같은 경우 과대적합이 의심 되기 때문에 배제하였다. (추후 데이터를 추가하여 분석할 예정)
# load_wine 데이터를 기준으로 평가할 때 FN보다 FP를 우선적으로 볼 때에도
# Precison이 0.95로 높은 Logistic Regression 알고리즘을 활용하여
# 모델을 구성하는 것이 적절한 것으로 판단된다.
