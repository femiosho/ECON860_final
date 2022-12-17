# Import all necessary libraries
import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the factor score dataset
data_predictor = pandas.read_csv("factor_score.csv")

# Change dataset to matrix format
predictor = data_predictor.values

# Load the original dataset
data_outcome = pandas.read_csv("dataset_final.csv")

# Extract the math column
outcome = data_outcome.iloc[:,40].values

# Split the predictor and outcome datasets into training and test sets
predictor_train, predictor_test, outcome_train, outcome_test = train_test_split(predictor, outcome, 
	test_size=0.3)

# Create and predict using the linear regression model
machine = linear_model.LinearRegression()
machine.fit(predictor_train, outcome_train)
prediction = machine.predict(predictor_test)

# Print linear regression prediction and R2 score
print("Linear Regression...........................")
print(f'Prediction using linear regression: {prediction}')
print(f'R2 score for linear regression: {metrics.r2_score(outcome_test, prediction)}')

# Create and predict using the logistic regression model
machine = linear_model.LogisticRegression()
machine.fit(predictor_train, outcome_train)
prediction = machine.predict(predictor_test)

# Print logistic regression prediction and R2 score
print()
print("Logistic Regression...........................")
print(f'Prediction using logistic regression: {prediction}')
print(f'R2 score for logistic regression: {metrics.r2_score(outcome_test, prediction)}')