# Import all necessary libraries
import pandas

# Load the dataset
all_questions = pandas.read_csv('dataset_no_zeroes.csv')

# Separate the math ability measure and the questionnaire responses
X = all_questions.drop('math', axis=1)
y = all_questions['math']

# Compute the Pearson correlation coefficient between math ability and each questionnaire response
correlations = X.corrwith(y, method='pearson')

# Sort the correlations by absolute value
correlations = correlations.abs().sort_values(ascending=False)

# Extract the top 20 questions
top_20_questions = correlations.index[:20]

# Print the top 20 questions
for i, question in enumerate(top_20_questions):
    print(f'Question {i+1}: {question}')