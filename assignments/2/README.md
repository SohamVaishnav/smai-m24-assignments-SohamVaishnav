# Assignment 2 Report

## Author: Soham Vaishnav
## Roll No.: 2022112002

### 2: Dataset
The dataset `word_embeddings.feather` is a collection of 200 words each encoded using 512 features, thereby rendering a very high dimensional data. On looking at it, the dataset can be broadly classified into verbs, inanimate and animate objects.

### 3: KMeans Clustering

#### Task 1: Implementation
The `class KMeansClustering()` is present in the folder `model\kmeans\kmeans.py` which includes functions such as `setK` (set the number of clusters), `getK` (obtain the number of clusters), `InitCentroids()` (initialises the centroids using **KMeans++** algo), `getCentroids()` (returns the final set of centroids), `fit()` (uses the EM algo to iteratively find the best set of centres and clusters the data around it), `predict()` (given a datapoint, it returns the cluster it is likely to belong to), `getCost()` (finds the cost incurred by the model using WCSS method).

- The **KMeans++** algo allows us to initialise the centroids in such a manner that a major portion of the dataset is spanned by those centroids due to large distance between them. This allows for a relatively better clustering (again depends on the dataset as well) 
- For fitting the model to the dataset, I followed and implemented the standard Expectation-Maximisation algorithm

#### Task 2: kmeans1
Here, we use the elbow method to find the most optimal number of clusters that the dataset can be divided into. Refer to the plots below for two datasets:
- `word_embeddings.feather`
- `data.csv` (provided via mail)

From the above plots, we find that best number of clusters for both datasets and from both models (self and sklearn) are as follows:

| Model : dataset | kmeans1 | WCSS |
|--|--|--|
| self : word | 6 | |
| sklearn : word | 6 | |
| self : data | 3 | |
| sklearn : data | 3 | |


### 4: Gaussian Mixture Models (GMMs)

#### Task 1: Implementation
The class `GaussianMixtureModel()` is present in the folder `model\gmm\gmm.py` and includes functions such as `getParams()` (returns the weights, covariance matrices and means of the gaussians), `getLikelihood()` (returns the log likelihood of the entire dataset), `getMembership()` (returns the degree to which each datapoint belongs to a particular gaussian), `InitParams()` (initialises the parameters to be used while fitting the model - more about this below), `fit()` (fits the model according to the data and returns the final set of parameters), `getClusters()` (returns the cluster that each datapoint belongs to) and `Gaussian()` (returns the multivariate gaussian pdf - however, not used).

- `getMembership()` uses the following equation:

$$
\gamma _{i, k} = \frac{\pi _{k} \mathcal{N}(x_i | \mu _k, \Sigma _k)}{\sum_{j=1}^{K} \pi _{j} \mathcal{N}(x_i | \mu _j, \Sigma _j)}
$$
