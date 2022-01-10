#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


breast_cancer = load_breast_cancer()


# In[3]:


# 변수확인
print(breast_cancer.DESCR)


# In[4]:


breast_cancer.keys()


# 데이터 이해하기

# In[5]:


#Feature Data 지정하기
breast_cancer_data = breast_cancer.data
breast_cancer_data.shape


# In[6]:


#label data 지정하기
label = breast_cancer.target
label.shape


# In[7]:


#target name 출력하기
target_name = breast_cancer.target_names
target_name


# In[9]:


import pandas as pd
breast_cancer_df = pd.DataFrame(data=breast_cancer_data, columns=breast_cancer.feature_names)
breast_cancer_df


# In[10]:


breast_cancer_df["label"] = breast_cancer.target
breast_cancer_df


# In[11]:


# train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data,
                                                    label,
                                                    test_size=0.2,
                                                    random_state=15)


# In[12]:


# 데이터셋 확인 
X_train.shape, y_train.shape


# In[13]:


X_test.shape, y_test.shape


# #### Decision Tree 사용하기

# In[14]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)# 모델 저장
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred))


# #### Random Forest 사용하기

# In[15]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# #### SVM 사용하기

# In[16]:


from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# #### SGD Classifier 사용하기

# In[17]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# #### Logistic Regression 사용하기

# In[18]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))


# ### 평가지표

# load_breast_cancer 데이터를 사용하여 총 5가지의 알고리즘을 바탕으로 분석하였다.
# 첫번째 알고리즘인 Decision Tree의 경우 accuracy는 0.95, Random Forest accuracy 0.94 ,
# SVM accuracy 0.87, SGD Classifier accuracy 0.81, Logistic Regression accuracy 0.89로 전반적으로 높은 정확도가 나타났다.
# 
# 총 5가지 알고리즘 중 sklearn.metrics 의 평가지표를 바탕으로 breast_cancer 데이터에 적절한 알고리즘으로는 Decision Tree을 뽑았다.
# 
# Decision Tree은 다른 알고리즘에 비해 accuracy이 가장 높기도 하지만
# load_breast_cancer 데이터를 기준으로 평가할 때 FP보다 FN를 우선적으로 볼 때에도
# Recall 0.93로 높기 때문이다.
# 그러므로 Decision Tree 알고리즘을 활용하여 모델을 구성하는 것이 적절한 것으로 판단된다.
