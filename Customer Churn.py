#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd

data = pd.read_csv(r'C:\Users\samar\Documents\Rutgers\Data Management for Data Science\Final Project\archive\data.csv')


# In[13]:


import psycopg2
from psycopg2 import sql


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)

data['MonthlyCharges'] = data['MonthlyCharges'].astype(float)

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="sam",
    host="localhost",
    port="5432"
)

cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customerID VARCHAR PRIMARY KEY,
        gender VARCHAR,
        SeniorCitizen INT,
        Partner VARCHAR,
        Dependents VARCHAR,
        tenure INT,
        PhoneService VARCHAR,
        MultipleLines VARCHAR,
        InternetService VARCHAR,
        OnlineSecurity VARCHAR,
        OnlineBackup VARCHAR,
        DeviceProtection VARCHAR,
        TechSupport VARCHAR,
        StreamingTV VARCHAR,
        StreamingMovies VARCHAR,
        Contract VARCHAR,
        PaperlessBilling VARCHAR,
        PaymentMethod VARCHAR,
        MonthlyCharges FLOAT,
        TotalCharges FLOAT,
        Churn VARCHAR
    )
""")
conn.commit()

for i, row in data.iterrows():
    cur.execute(sql.SQL("""
        INSERT INTO customers (
            customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """), (
        row['customerID'], row['gender'], row['SeniorCitizen'], row['Partner'], row['Dependents'],
        row['tenure'], row['PhoneService'], row['MultipleLines'], row['InternetService'],
        row['OnlineSecurity'], row['OnlineBackup'], row['DeviceProtection'], row['TechSupport'],
        row['StreamingTV'], row['StreamingMovies'], row['Contract'], row['PaperlessBilling'],
        row['PaymentMethod'], row['MonthlyCharges'], row['TotalCharges'], row['Churn']
    ))
conn.commit()

cur.close()
conn.close()


# In[15]:


data = data.fillna(method='ffill')

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(0)


# In[18]:


data['total_spent'] = data['tenure'] * data['MonthlyCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['tenure', 'MonthlyCharges', 'TotalCharges']])


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data['tenure'], kde=True)
plt.show()


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

data['TotalSpent'] = data['tenure'] * data['MonthlyCharges']

features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalSpent', 'SeniorCitizen', 'Partner', 'Dependents']
X = pd.get_dummies(data[features], drop_first=True)
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[29]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC Score: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")


# In[30]:


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalSpent', 'SeniorCitizen', 'Partner', 'Dependents']])
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)

