# The contents of the file: /stock-price-prediction/stock-price-prediction/price.py

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # type: ignore

# Data Preprocessing
df = pd.read_csv('stock_price_direction_data.csv')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df.drop(columns=['Date'], inplace=True)

y = df['Target']
x = df.drop(columns=['Target'])

x.fillna(x.mean(), inplace=True)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)

# SVM
model1 = SVC(kernel='rbf', random_state=42)
model1.fit(xtrain_scaled, ytrain)
model1_predictions = model1.predict(xtest_scaled)

print('SVM classification report:')
print(classification_report(ytest, model1_predictions))
print('SVM Accuracy score:')
print(accuracy_score(ytest, model1_predictions))
model1_cm = confusion_matrix(ytest, model1_predictions)

# KNN
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(xtrain_scaled, ytrain)
model2_predictions = model2.predict(xtest_scaled)

print('KNN classification report:')
print(classification_report(ytest, model2_predictions))
print('Accuracy score:')
print(accuracy_score(ytest, model2_predictions))
model2_cm = confusion_matrix(ytest, model2_predictions)

# Visualizations
# Confusion matrix visualization
# SVM
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(model1_cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].set_title('SVM Confusion Matrix')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Down', 'Up'])
axes[0].set_yticklabels(['Down', 'Up'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, model1_cm[i, j], ha='center', va='center', color='red')

# KNN
axes[1].imshow(model2_cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title("KNN Confusion Matrix")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Down', 'Up'])
axes[1].set_yticklabels(['Down', 'Up'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, model2_cm[i, j], ha='center', va='center', color='red')

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='0.2f')
plt.title('Correlation Heatmap')
plt.show()

# Closing price over time using line plot
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Close price', color='blue')
plt.title('Closing price over time')
plt.xlabel('Index')
plt.ylabel('Closing time')
plt.legend()
plt.show()

# Target Distribution using count plot
plt.figure(figsize=(10, 6))
sns.countplot(x=y, hue=y, palette='viridis', legend=False)
plt.title('Target variable distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()