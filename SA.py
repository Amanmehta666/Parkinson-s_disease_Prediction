# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the dataset
parkinsons_data = pd.read_csv('pdd.csv')

# Data preprocessing
X = parkinsons_data.drop(['status'], axis=1)
y = parkinsons_data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build the models
models = {'Logistic Regression': LogisticRegression(),
          'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators=500, max_features='sqrt')}

# Train the models and evaluate their performance
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy of the predictions
    acc = accuracy_score(y_test, y_pred)
    
    # Print the classification report and confusion matrix
    print('Classification Report for', model_name, ':')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix for', model_name, ':')
    print(confusion_matrix(y_test, y_pred))
    
    # Store the results in a dictionary
    results[model_name] = acc

# Display a correlation matrix using a seaborn visualization
numeric_cols = X.select_dtypes(include=np.number).columns
corr = parkinsons_data[numeric_cols].corr()
sns.heatmap(corr, annot=True)

# Display the output in a bar chart and a pie chart using a GUI
root = tk.Tk()
root.title('Parkinson\'s Disease Prediction')

# Create a figure and axis for the bar chart
fig1 = plt.figure(figsize=(8,6))
ax1 = fig1.add_subplot(111)

# Add a bar chart to the axis
ax1.bar(results.keys(), results.values())

# Add labels to the bar chart
ax1.set_xlabel('Algorithms')
ax1.set_ylabel('Accuracy')
ax1.set_title('Parkinson\'s Disease Prediction')

# Add a legend to the bar chart
ax1.legend()

# Display the output in a GUI form
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.draw()
canvas1.get_tk_widget().pack()

# Create a new window for the pie chart
root2 = tk.Toplevel()
root2.title('Parkinson\'s Disease Distribution')

# Create a figure and axis for the pie chart
fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.add_subplot(111)

# Add a pie chart to the axis
counts = y_test.value_counts()
ax2.pie(counts, labels=counts.index, autopct='%1.1f%%')

# Add a title to the pie chart
ax2.set_title('Parkinson\'s Disease Distribution')

# Display the output in a GUI form
canvas2 = FigureCanvasTkAgg(fig2, master=root2)
canvas2.draw()
canvas2.get_tk_widget().pack()

root.mainloop()
