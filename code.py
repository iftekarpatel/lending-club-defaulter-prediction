# --------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(filepath_or_buffer=path,compression = 'zip',low_memory = False)
X = df.drop('loan_status',1)
y = df['loan_status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 4)

# Code ends here


# --------------
# Code starts  here
col = df.isnull().sum()
col_drop = df.columns[df.isnull().sum()*100/len(df) > 25]
col_drop = df.columns[df.nunique() == 1].append(col_drop)
X_train.drop(col_drop,1,inplace = True)
X_test.drop(col_drop,1,inplace = True)

# Code ends here


# --------------
import numpy as np


# Code starts here
y_train =np.where((y_train == 'Fully Paid') |(y_train == 'Current'), 0, 1)
y_test = np.where((y_test == 'Fully Paid') |(y_test == 'Current'), 0, 1)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder


# categorical and numerical variables
cat = X_train.select_dtypes(include = 'O').columns.tolist()
num = X_train.select_dtypes(exclude = 'O').columns.tolist()

# Code starts here
for c in num:
    X_train[c].fillna(X_train[c].mean(),inplace = True)
    X_test[c].fillna(X_test[c].mean(),inplace = True)
for c in cat:
    X_train[c].fillna(X_train[c].mode()[0],inplace = True)
    X_test[c].fillna(X_test[c].mode()[0],inplace = True)
for c in cat:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c])
    X_test[c] = le.fit_transform(X_test[c])
 
# Code ends here


# --------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix,classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

# Code starts here
rf = RandomForestClassifier(random_state =42,max_depth=2,min_samples_leaf=5000)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
f1 = f1_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
y_pred_proba = rf.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label="Random Forest model, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# Code ends here


# --------------
from xgboost import XGBClassifier

# Code starts here
xgb = XGBClassifier(learning_rate = 0.0001)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
f1 = f1_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
y_pred_proba = xgb.predict_proba(X_test)[:,1]
print(y_pred_proba)
from sklearn.metrics import roc_curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label="XGBoost model, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# Code ends here


