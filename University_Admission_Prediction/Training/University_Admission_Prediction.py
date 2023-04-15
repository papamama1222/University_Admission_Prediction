import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam


# Load the data
data = pd.read_csv('D:/University_Admission_Prediction/Dataset/Admission_Predict.xls')
print(data.info())
print(data.isnull().any())

# Rename the column
data=data.rename(columns = {'Chance of Admit ':'Chance of Admit'})

print(data.describe())
# Plot the distribution of GRE scores
sns.histplot(data['GRE Score'])
plt.title('GRE Score')
plt.show()

sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Create a pair plot to visualize relationships between different features, based on whether the student has research experience or not
sns.pairplot(data=data,hue='Research',markers=["^","v"],palette='inferno')
plt.title('a')
plt.show()

# Create a scatter plot to visualize the relationship between University Rating and CGPA
sns.scatterplot(x='University Rating',y='CGPA',data=data,color='Red',s=100)
plt.title('scatter plot')
plt.show()



# Rename the column
data = data.rename(columns={'Chance of Admit ':'Chance of Admit'})

# Define the categories and colors
categories = data.columns[1:]  # Exclude the first column
colors = ['yellowgreen', 'gold','lightskyblue','pink','red','purple','orange','gray']

# Plot histograms for each pair of categories
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(14, 9))
for i, ax in enumerate(axs.flatten()):
    if i < len(categories):
        ax.hist(data[categories[i]], color=colors[i%8], bins=10)
        ax.set_title(f'Category {i+1}: {categories[i]}')
        ax.set_xlabel('')
    else:
        fig.delaxes(ax)
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()



# Split the data into features (X) and target variable (y)
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Scale the features using MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Convert the y_train and y_test variables into binary labels
y_train = (y_train > 0.5)
y_test = (y_test > 0.5)

# Build the logistic regression model
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_lr = lr.predict(X_test)

# Build the ANN model
model = keras.Sequential()
model.add(Dense(7, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# Compile the model
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

from sklearn.metrics import classification_report

lr.fit(X_train, y_train)

train_predictions = lr.predict(X_train)
print(train_predictions)

train_acc = lr.score(X_train, y_train)
print(train_acc)

test_acc = lr.score(X_test, y_test)
print(test_acc)

pred = lr.predict(X_test)

classification_report(y_test, pred, zero_division=1)



from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix


# Logistic Regression Evaluation Metrics
y_pred_lr = lr.predict(X_test)
print("\n Logistic Regression Evaluation Metrics:")
print(" Accuracy score: %f" %(accuracy_score(y_test,y_pred_lr) * 100))
print(" Recall score : %f" %(recall_score(y_test,y_pred_lr) * 100))
print(" ROC score : %f\n" %(roc_auc_score(y_test,y_pred_lr) * 100))
print(confusion_matrix(y_test,y_pred_lr))
print(classification_report(y_test, y_pred_lr, zero_division=1))


# ANN Evaluation Metrics

y_pred_ann = model.predict(X_test)
y_pred_ann = (y_pred_ann > 0.5)
print("\n ANN Evaluation Metrics:")
print(classification_report(y_test,y_pred_ann, zero_division=1))

y_pred_ann_train = model.predict(X_train)
y_pred_ann_train = (y_pred_ann_train > 0.5)
print("\n ANN Training Evaluation Metrics:")
print(classification_report(y_train,y_pred_ann_train, zero_division=1))

model.save('model.h5')

