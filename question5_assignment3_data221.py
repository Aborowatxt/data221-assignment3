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


# List of k values to test
k_values = [1, 3, 5, 7, 9]

# Store each model's test accuracy
accuracies = []

# Train and evaluate model for each k
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k) #Creates a KNN model for each k
    model.fit(X_train, y_train)  #Trains each model using the training data

    y_predictions = model.predict(X_test)    #model predictions
    k_accuracy = accuracy_score(y_test, y_predictions)   #Checks how many models were correct

    accuracies.append(k_accuracy)   #adds accuracy value to existing list

# use pandas to create a table dataframe
results = pd.DataFrame({"k": k_values,"Test Accuracy": accuracies})

print("Accuracy for different k values:")
print(results)

#finds which row has the highest accuracy.
best_k = results.loc[results["Test Accuracy"].idxmax(), "k"]
print("Best k value:", best_k)


"""
As k gets smaller, the model becomes more sensitive to individual data points, which can cause overfitting. When k is very 
small, the model can basically start memorizing noise instead of actually learning real patterns. As k gets bigger, the 
model becomes smoother and less sensitive to specific points, which can lead to underfitting. If k is too large, the model 
might oversimplify the data and miss important details. So, choosing the right k is really about finding a balance between 
overfitting and underfitting.

"""

