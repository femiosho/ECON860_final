# Import all necessary libraries
import pandas

# Load the dataset
factors = pandas.read_csv('factor_loadings.csv')

# Calculate the average for each row
factors_averages = factors.abs().mean(axis=1)

# Sort the row averages from highest to lowest
factors_sorted_averages = factors_averages.sort_values(ascending=False)

# Print the sorted row averages
# print(factors_sorted_averages)

# Extract the top 20 questions
top_20_questions = factors_sorted_averages.index[:20]

# Print the top 20 questions
for i, question in enumerate(top_20_questions):
    print(f'Question {i+1}: Q{question+1}')