# Import all necessary libraries
import pandas
import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load the factor score dataset
dataset  = pandas.read_csv("factor_score.csv")

# Change dataset to matrix format
dataset = dataset.values

# K-Means function definition
def run_kmeans(n, dataset):
  machine = KMeans(n_clusters=n, n_init='auto')
  machine.fit(dataset)
  results = machine.predict(dataset)
  centroids = machine.cluster_centers_
  silhouette = silhouette_score(dataset, results, metric = "euclidean")
  # Visualization was restricted to the first two columns because
  # it was not possible to visualize the five columns
  pyplot.scatter(dataset[:,0], dataset[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:,1], c='red', s=100)
  pyplot.savefig("scatterplot_kmeans_" + str(n) + ".png")
  pyplot.close()
  return silhouette

# K-Means function call
silhouette_result = run_kmeans(5, dataset)
print(f'K-Means silhouette score: {silhouette_result}')

# Gaussian mixture model function definition
def run_gmm(n, dataset):
  machine = GaussianMixture(n_components=n)
  machine.fit(dataset)
  results = machine.predict(dataset)
  centroids = machine.means_
  silhouette = silhouette_score(dataset, results, metric = "euclidean")
  # Visualization was restricted to the first two columns because
  # it was not possible to visualize the five columns
  pyplot.scatter(dataset[:,0], dataset[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:,1], c='red', s=100)
  pyplot.savefig("scatterplot_gmm_" + str(n) + ".png")
  pyplot.close()
  return silhouette

# GMM function call 
silhouette_result = run_gmm(5, dataset)
print(f'GMM silhouette score: {silhouette_result}')