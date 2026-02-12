import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


#Load the kidney dataset into pandas dataframe
kidney_disease_dataframe = pd.read_csv("kidney_disease.csv")
print(kidney_disease_dataframe.columns)

# Separate features (X) and target label (y)
X = kidney_disease_dataframe.drop("classification", axis=1)


# KNN requires numeric input, so this prevents "could not convert string to float" errors
X = pd.get_dummies(X)
# KNN cannot handle NaN values, so this prevents "Input contains NaN" errors
X = X.fillna(X.median(numeric_only=True))

# Clean and convert classification labels to numeric values (0 and 1)
y = kidney_disease_dataframe["classification"].astype(str).str.strip()
y = y.map({"notckd": 0, "ckd": 1})

#Split the data into 70% training and 30% testing sets using a fixed random_state to ensure reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Creating the model and setting k to 5
knn = KNeighborsClassifier(n_neighbors=5)

#Training the model using training data
knn.fit(X_train, y_train)

#Making predictions on the test data
y_predictions = knn.predict(X_test)

# Generate confusion matrix to see breakdown of predictions
cm = confusion_matrix(y_test, y_predictions)
print("Confusion Matrix:")
print(cm)

# Calculate overall accuracy (percentage of correct predictions)
print("Accuracy:", accuracy_score(y_test, y_predictions))

# Precision: of predicted CKD cases, how many were correct
print("Precision:", precision_score(y_test, y_predictions))

# Recall: of actual CKD cases, how many were correctly identified
print("Recall:", recall_score(y_test, y_predictions))

# F1 Score: harmonic mean of precision and recall
print("F1 Score:", f1_score(y_test, y_predictions))

"""
In the context of the kidney disease prediction, True positive and true negative represents the number of patients correctly
identifies as having the kidney disease. While the false positive is when a healthy patients is identifies as having the disease and 
false negative is when a patient with the kidney disease is not identifies as having the disease. Accuracy alone may not be enough to
evaluate a model because it doesn't show how well the model performs on each class individually, just the overall proportion of the 
correct prediction. Unlike other measures like recall and precision that measure the reliability and how well the model detects diseases.
If missing a case is very serious, recall is most important, because it minimizes false negatives. In other words it focuses
specifically on the missed cases.
"""
