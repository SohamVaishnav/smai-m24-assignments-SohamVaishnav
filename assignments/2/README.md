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

- memberships, likelihood and parameters have been calculated according to the slides provided in class
- `InitParams()` has `init_method (str)` as an input argument which determines the way the parameters will be initialised (especially covariance matrix)
    - If `init_method = "random"`, then we randomly select the values for the parameters
    - If `init_method = "random_from_data"`, then we randomly select the values for mean from the dataset (any k datapoints), cov matrix becomes the diagonal matrix with variance of dataset and weights are uniform
    - If `init_method = "identity"`, this only applies to the covariance matrix which is initialised as identity matrix 
- `fit()` uses the EM algo to converge over user-defined epochs
    - Also uses `epsilon` as an input which prevents the denominator from going to 0 while E or M step and also regularises the covariance matrix if its values become too small. The convergence of the dataset sometimes depends on the value of epsilon

#### Task 2: Analysis, AIC/BIC and kgmm1
Here, we run the GMM over the provided 512 dimensional dataset. The GMM used here for analysis is that built by me as well as the one provided by sklearn. Following is the comparative analysis of the performance of both over some factors of the GMM:

- self-built GMM:

| k | Likelihood | Means | 
|--|--|--|
| 1 |  |  |
| 2 |  |  |
| 5 |  |  |
| 7 |  |  |
| 9 |  |  |
| 10 |  |  |

- sklearn GMM:

| k | Likelihood | Means | 
|--|--|--|
| 1 |  |  |
| 2 |  |  |
| 5 |  |  |
| 7 |  |  |
| 9 |  |  |
| 10 |  |  |

Inference:
- Both the GMMs fail on the given 512 dimensional dataset - because they produce a positive log-likelihood (probability can never be > 1!). There are two reasons that I can think of:
    - The cov matrix over a high dimensional data usually tends to become singular, particularly when
    the number of features exceeds the number of datapoints. This reduces the rank of the matrix because many features are linear combination of other features thereby becoming taxing on the model. 
    - The choice of a good regularisation parameter becomes quite crucial because that by and large affects the convergence of the model 
    - Also, the way the parameters are initialised plays a role in the convergence of the model. However, I tried to take care of that as much as possible by introducing similar initial states to the sklearn GMM as I did for my class

- Both GMMs do pretty well on the `data.csv` dataset and exactly due to the reasons mentioned above. More impressive is the fact that the self-built model and the sklearn model both output nearly same parameters!

For AIC and BIC, we find the following plots:

From the above plots, we infer that the optimal number of clusters for the dataset provided is _kgmm1_ = ENTER THE VALUE.


### 5: Dimensionality Reduction and Visualisation

#### Task 1: Implementation
The class `PCA()` is present in the folder `models\pca\pca.py` and includes functions such as `getComponents()` (returns the number of components the dataset is being reduced to), `getEigenvalues()` (returns the eigenvalues of the covariance matrix), `getEigenvectors()` (returns the eigenvectors of the covariance matrix), `fit()` (finds the top k eigenvectors corresponding to the max k eigenvalues), `transform()` (transforms the original dataset by projecting it on the principal components) and `checkPCA()` (asserts whether the data has actually been reduced or not).

- Eigenvalues and eigenvectors are calculated for the covariance matrix of the dataset. The eigenvectors are a set of orthogonal vectors. 
- `fit()` selects the k eigenvectors by sorting the corresponding eigenvalues in descending order and choosing the first k out of those. 
- `transform()` takes the final k components obtained after the `fit()` step and takes the dot product of the dataset with those components thereby creating the projections of the dataset in the space spanned by the k eigenvectors.  
- `checkPCA()` performs the inverse PCA over the reduced dataset to try and obtain the original dataset as accurately as possible. 
    - this basically involves taking dot product of reduced dataset with the transpose of the matrix containing k eigenvectors. 
    - above method stems from the fact that the matrix containing eigenvectors is orthogonal which means that its dot product with its own transpose gives us an identity matrix. 

#### Task 2: Testing and Visualisation



#### Task 3: Data Analysis
The axes obtained from PCA are the eigenvectors with maximum eigenvalues of the covariance matrix of the dataset. What this means is that the axes represent the directions where, on projecting, the variance of the dataset is maximised. This helps in finding patters in the dataset which would otherwise have been suppressed in the original dimensions.

From the 2D and 3D plots shown above, we find that the optimal number of clusters for the dataset, referred to as _k2_ are _____ (FILL HERE).


### 6: PCA + Clustering

#### Task 1: KMeans with k = _k2_
Results after performing KMeans clustering on the dataset for k = _k2_ (obtained from the 2D plots in PCA).

#### Task 2: PCA + KMeans
Following is the **scree plot** for the dataset:
ENTER THE PLOT

The optimal number of dimensions that the dataset should be reduced to is _____ (FILL HERE). The _reduced dataset_ therefore contains _____ features.

Elbow plot for the reduced dataset:
ENTER THE PLOT

From the above plot, we find that optimal number of clusters for reduced dataset, referred to as _kmeans3_ is _____ (FILL HERE).

After performing KMeans on the reduced dataset for k = _kmeans3_, we find that the following cluster plot:
ENTER PLOT HERE

#### Task 3: GMM with k = _k2_
Results after performing GMM on the dataset for k = _k2_ (obtained from the 2D plots in PCA).

#### Task 4: PCA + GMMs
Optimal number of clusters for the reduced dataset based on AIC or BIC, referred to as _kgmm3_ is _____ (FILL HERE).
The associated plot is as follows:
ENTER THE PLOT

The clusters formed after applying GMM on the reduced dataset for k = _kgmm3_ is as follows:
ENTER THE PLOT


### 7: Cluster Analysis

#### Task 1: KMeans cluster analysis


#### Task 2: GMM cluster analysis

#### Task 3: Compare KMeans and GMM


### 8: Hierarchical Clustering


### 9: Nearest Neighbor Search 

