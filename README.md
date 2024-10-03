# classification-challenge
classification-challenge

Spam Detector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Retrieve the Data
The data is located at https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv

Dataset Source: UCI Machine Learning Library

Import the data using Pandas. Display the resulting DataFrame to confirm the import was successful.

# Import the data
data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv")
data.head()

Predict Model Performance
You will be creating and comparing two models on this data: a Logistic Regression, and a Random Forests Classifier. Before you create, fit, and score the models, make a prediction as to which model you think will perform better. You do not need to be correct!

Write down your prediction in the designated cells in your Jupyter Notebook, and provide justification for your educated guess.

If there is a linear relationship, Logistic Regression may perform better. I think because there might be either or in the data.


#Split the Data into Training and Testing Sets

Create the labels set `y` and features DataFrame `X`
Create the labels set y
y = data['spam']

Create the features DataFrame X
X = data.drop('spam', axis=1)

Check the balance of the labels variable (`y`) by using the `value_counts` function.
y.value_counts()

Split the data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_size=0.2 means that 20% of the data will be used for testing and the rest for training.
random_state=42 is used to ensure reproducibility of the results.


Print the shape of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Scale the Features
Use the StandardScaler to scale the features data. Remember that only X_train and X_test DataFrames should be scaled.


from sklearn.preprocessing import StandardScaler

Create the StandardScaler instance
scaler = StandardScaler()

Fit the scaler to the training data and transform both the training and testing data

Fit the Standard Scaler with the training data
scaler.fit(X_train)

Transform both the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create and Fit a Logistic Regression Model
Create a Logistic Regression model, fit it to the training data, make predictions with the testing data, and print the model's accuracy score. You may choose any starting settings you like.

Train a Logistic Regression model and print the model score
from sklearn.linear_model import LogisticRegression

Make and save testing predictions with the saved logistic regression model using the test data
Create and train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

Make predictions on the scaled test data
predictions = model.predict(X_test_scaled)

Create a DataFrame with the predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

Save the predictions DataFrame to a CSV file
predictions_df.to_csv('predictions.csv', index=False)

Review the predictions
print(predictions_df)

Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
Calculate the accuracy score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)

#Create and Fit a Random Forest Classifier Model
Create a Random Forest Classifier model, fit it to the training data, make predictions with the testing data, and print the model's accuracy score. You may choose any starting settings you like.


Train a Random Forest Classifier model and print the model score
from sklearn.ensemble import RandomForestClassifier
model_random_forest = RandomForestClassifier()
model_random_forest.fit(X_train_scaled, y_train)

RandomForestClassifier()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

Make and save testing predictions with the saved logistic regression model using the test data
predictions = model_random_forest.predict(X_test_scaled)
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
predictions_df.to_csv('predictions.csv', index=False)
Review the predictions
print(predictions_df)

Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
from sklearn.metrics import accuracy_score

Calculate the accuracy score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy Score:", accuracy)


#Evaluate the Models
Which model performed better? How does that compare to your prediction? Write down your results and thoughts in the following markdown cell.

It looks like the Random Forest Classifier performed better with 0.957 compared to the Logistic Regression model's score of 0.919. I thought this would be a more linear data set being incorrect. The Random Forest might be able to capture more complex patterns where linear might not see that in analysis.






