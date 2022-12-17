# Import all necessary libraries
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load the dataset
df = pd.read_csv("dataset_final.csv")

# Replace all zeros with NaN values
df = df.replace(0, np.nan)

# Create an instance of IterativeImputer with values ranging from 1 to 5
imputer = IterativeImputer(max_iter=10, random_state=0, min_value=1, max_value=5)

# Fit the imputer to the dataset
imputer.fit(df)

# Use the imputer to transform the dataset and replace the missing values
df_no_zeroes = imputer.transform(df)

# Convert the imputed data back to a DataFrame
df_no_zeroes = pd.DataFrame(df_no_zeroes, columns=df.columns)

# Convert the column to integers
df_no_zeroes = df_no_zeroes.round().astype(int)

# save the modified DataFrame back to a CSV file
df_no_zeroes.to_csv("dataset_no_zeroes.csv", index=False)