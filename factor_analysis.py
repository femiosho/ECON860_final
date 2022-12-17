# Import all necessary libraries
import pandas
from factor_analyzer import FactorAnalyzer
import numpy
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

# Load the dataset with no zeroes
dataset = pandas.read_csv("dataset_no_zeroes.csv")

# Extract the 40-question data
dataset_subset = dataset.iloc[:,:40]

# Get the Bartlett test of sphericity
chi2,p=calculate_bartlett_sphericity(dataset_subset)
# print(chi2, p)

# Get the eigenvalues
machine = FactorAnalyzer(n_factors=40, rotation=None)
machine.fit(dataset_subset)
ev, v = machine.get_eigenvalues()
# print(ev)

# Based on the eignevalues >= 1, we extracted 7 factors.
# However, after testing 5, 6, and 7 factors using 
# oblimin (an oblique rotation method) and varimax
# (an orthogonal rotation method), the 5-factor (in both 
# method cases) had the lowest number of cases of 
# cross-loadings and low loadings
machine = FactorAnalyzer(n_factors=5, rotation='varimax')
machine.fit(dataset_subset)
factor_loadings = machine.loadings_

# Round the factor loadings to 2 decimal places
# print(factor_loadings.round(3))

# Save factor loadings to csv file
pandas.DataFrame(factor_loadings).to_csv("factor_loadings.csv", 
	index=False, header=['F1', 'F2', 'F3', 'F4', 'F5'])

# Change dataset to matrix format
dataset_subset = dataset_subset.values

# Get the factor scores for each participant
factor_score = numpy.dot(dataset_subset, factor_loadings)

# Save factor scores to csv file
pandas.DataFrame(factor_score).round().to_csv("factor_score.csv", 
	index=False, header=['F1', 'F2', 'F3', 'F4', 'F5'])