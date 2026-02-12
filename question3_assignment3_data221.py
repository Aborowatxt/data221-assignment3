import pandas as pd
from sklearn.model_selection import train_test_split


#Load the kidney dataset into pandas dataframe
kidney_disease_dataframe = pd.read_csv("kidney_disease.csv")
print(kidney_disease_dataframe.columns)

# Separate features (X) and target label (y)
X = kidney_disease_dataframe.drop("classification", axis=1)
y = kidney_disease_dataframe["classification"]

#Split the data into 70% training and 30% testing sets using a fixed random_state to ensure reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print shapes to confirm correct 70/30 split
print(X_train.shape)
print(X_test.shape)


"""
Why we should not train and test a model on the same data
If we did this then the model would just memorize the data, and nothing will be left for prediction. if the model is
based on one data, then it assumes the results of that data for every other kidney datasets out there. Leads to 
overfitting

What the purpose of the testing set is
The testing set simulates real-world scenarios by evaluating the model on data it has never seen before. 
This helps measure how well the model generalizes and whether it makes accurate predictions on new patients.
"""