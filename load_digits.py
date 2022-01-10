#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[3]:


# 데이터 준비
digits = load_digits()


# In[5]:


digits.keys()


# # 데이터 이해하기

# In[12]:


#Feature Data 지정하기
digits_data = digits.data
digits_data.shape


# In[8]:


#label data 지정하기
label = digits.target
label.shape


# In[10]:


#target name 출력하기
target_name = digits.target_names
target_name


# In[11]:


# 데이터 describe해보기
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(digits.data[0].reshape(8, 8), cmap='gray')
plt.axis('off')
plt.show()


# In[13]:


# train, test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(digits_data,
                                                    label,
                                                    test_size=0.2,
                                                    random_state=15)


# In[14]:


# 데이터셋 확인 
X_train.shape, y_train.shape


# In[15]:


X_test.shape, y_test.shape


# # 모델 학습 하기
# 

# #### Decision Tree 사용하기

# In[17]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(random_state=32)# 모델 저장
decision_tree.fit(X_train, y_train)


# In[18]:


y_pred = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred))


# #### Random Forest 사용하기

# In[19]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))


# #### SVM 사용하기

# In[20]:


from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))


# #### SGD Classifier 사용하기

# In[21]:


from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier()

sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))


# #### Logistic Regression 사용하기

# In[22]:


from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))


# ### 모델 평가하기

# In[ ]:


load_digits 데이터를 사용하여 총 5가지의 알고리즘을 바탕으로 분석하였다.
첫번째 알고리즘인 Decision Tree의 경우 accuracy는 0.84, Random Forest accuracy 0.98 ,
SVM accuracy 0.98, SGD Classifier accuracy 0.94, Logistic Regression accuracy 0.97로
전반적으로 높은 정확도가 나타났다.

총 5가지 알고리즘 중 sklearn.metrics 의 평가지표를 바탕으로 digits 데이터에 적절한 
알고리즘으로는 Random Forest와 SVM을 뽑았다.

Random Forest와 SVM은 다른 알고리즘에 비해 accuracy이 가장 높기도 했지만
load_digits 데이터를 기준으로 평가할 때 FN보다 FP를 우선적으로 보기 때문에
accuracy도 높지만 Precison이 0.98로 높은 Random Forest와 SVM 알고리즘을 활용하여
모델을 구성하는 것이 적절한 것으로 판단된다.

