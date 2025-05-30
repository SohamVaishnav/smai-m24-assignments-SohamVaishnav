# Assignment 2 Report

## Author: Soham Vaishnav
## Roll No.: 2022112002

### 2: Multi Layer Perceptron Classification

#### Task 1: Data Analysis and Preprocessing
The dataset `WineQT.csv` is a dataset of wine quality ratings. It contains 13 features and a target variable `quality` which ranges from 3 to 8. Following is the visualisation of the features of the dataset, their internal relationships and the dependence of the target variable on the features.

![WineQT Pair Plot](figures/PairPlot_WineQT.png)

Observations:
- The features `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density` and `pH` have a roughly exponential relationship with the target variable.
- The features `alcohol` and `sulphates` have a roughly linear relationship with the target variable.
- The dataset is imbalanced with respect to the target variable - there are more samples for lower quality wines and fewer samples for higher quality wines.
- The data is quite congested in the lower half of the feature space, thereby, making classification a challenging task. 

For preprocessing, I have used the `DataPreprocess()` function in `a3.py` which performs the following tasks:
- Checks for null values and removes them.
- Standardises the data so that the mean of each feature is 0 and the standard deviation is 1.
- Splits the data into training and testing sets (80-20 random split).

#### Task 2: Model Building from Scratch
The model has been implemented in `models/MLP/mlp.py`, the features of which are as follows:
- The model has functions such as `add()` (adds a layer to the model), `fit()` (initialises weights and biases and fits the model to the training data), `train()` (trains the model on the training data), `predict()` (predicts the target variable for the test data) and `evaluate()` (evaluates the model on the test data), `loss()` (calculates the loss between the predicted and actual values) and `backprop()` (calculates the gradients of the loss function with respect to the weights and biases), `forward()` (performs a forward pass through the network). 
- The model also uses the `Optimizer` that includes functions such as `sgd()` (stochastic gradient descent), `bgd()` (batch gradient descent) and `mini_bgd()` (mini-batch gradient descent).
- Another helper class `Layers()` is used to store the architecture of the network such as the number of units in each layer and the activation function to be used. 
- The loss function used is cross-entropy and the activation function used in the last layer is softmax.
- To initialise the model, a config file is used which contains all the hyperparameters of the model.
- Weights are initialised using the Xavier method and biases are initialised to 0.
- Inside the `MultiLayerPerceptron_SingleClass()` class, a numerical gradient method has also been implemented to check the analytical gradients. The validity of gradients calculated by backprop method is checked using the norm of the gradients calculated analytically and numerically.

#### Task 3: Hyperparameter Tuning using W&B
Following are the results of the hyperparameter tuning:

![WandB_1_mlp_single](figures/WandB_1_mlp_single.png)

![WandB_2_mlp_single](figures/WandB_2_mlp_single.png)

The sweep results based on the above plots are:

![Sweep_mlp_single](figures/Sweep_mlp_single.png)

Based on the above results, the best set of hyperparameters are:

|Learning Rate|Epochs|Batch Size|Optimizer|Hidden Layers|Activation Functions|
|:-----------:|:----:|:--------:|:-------:|:------------:|:-------------------:|
|0.01|50|32|sgd|Three - [32, 16, 6]|[relu, relu, softmax]|

A table containing the HPT results has been put under the `data\internals\3` folder (it is a little large to be displayed here).

#### Task 4: Evaluation of best model
Using the best set of hyperparameters, the model has been evaluated on the valid and test datas. The results are as follows:

For validation set:

|Loss|Accuracy|Precision|Recall|F1 Score|
|:---:|:------:|:-------:|:----:|:------:|
|0.80109|0.66667|0.30242|0.29478|0.29856|

For test set:

|Loss|Accuracy|Precision|Recall|F1 Score|
|:---:|:------:|:-------:|:----:|:------:|
|1.00648|0.56522|0.29352|0.27518|0.28406|

#### Task 5: Analysing hyperparameter effects

##### Task 5.1: Effect of Non-linearity

Validation Loss vs Epochs
![Effect_Non_Linearity_mlp_single](figures/effect_non_linearity_mlp_single.png)

Training Loss vs Epochs
![Effect_Non_Linearity_Training_mlp_single](figures/effect_non_linearity_mlp_single_train.png)

##### Task 5.2: Effect of Learning Rate

Training Loss vs Epochs
![Effect_Learning_Rate_mlp_single](figures/effect_lr_mlp_single_train.png)

Validation Loss vs Epochs
![Effect_Learning_Rate_mlp_single](figures/effect_lr_mlp_single.png)

##### Task 5.3: Effect of Batch Size

Training Loss vs Epochs
![Effect_Batch_Size_mlp_single_train](figures/effect_bs_mlp_single_train.png)

Validation Loss vs Epochs
![Effect_Batch_Size_mlp_single_valid](figures/effect_bs_mlp_single.png)

Observations:
- We see that the model is quite sensitive to the learning rate. A learning rate that is too high causes the model to diverge and also overfit. Whereas, a learning rate that is too low causes the model to converge too slowly.
- The model is quite sensitive to the activation functions used - using relu as the activation function for the hidden layers and softmax in the last layer gives the best results. Sigmoid convergence is the slowest and tanh is faster than relu.
- The model is also sensitive to the batch size - using a batch size of 1 is equivalent to using stochastic gradient descent and using a batch size equal to the training set size is equivalent to using batch gradient descent. Using a mini-batch gradient descent with an optimal batch size gives faster convergence than using stochastic gradient descent. However, on the downside, the change in loss is not much significant for each epoch. 

#### Task 6: Multi-label Classification
The dataset used here is `advertisement.csv` which contains the information of some 1000 individuals and the kinds of products they have purchased as well as the categories of the products that highlights the kind of advertisement they are most likely to click on.

Here is a visualisation of the multiple features of the dataset:
![Age_Ads](figures/Age_Ads.png) 
![Bought_items_Ads](figures/Bought_items_Ads.png)
![City_Edu_Items_Ads](figures/City_Edu_Items_Ads.png) 
![City_Occ_Items_Ads](figures/City_Occ_Items_Ads.png) 
![City_wrt_Gender_Ads](figures/City_wrt_Gender_Ads.png) 
![City_wrt_Items_Ads](figures/City_wrt_Items_Ads.png) 
![Gender_Marriage_Ads](figures/Gender_Marriage_Ads.png)
![Items_wrt_Gender_Ads](figures/Items_wrt_Gender_Ads.png)

The above plots present the demographic information of the individuals and the kind of products they have purchased. They also show the dependence of items bought (and therefore the ads seen) on the gender of the individual, their education and occupation.

For the model, I have used the `MultiLayerPerceptron_MultiClass()` class in `models/MLP/mlp_multi.py` which is similar to the `MultiLayerPerceptron_SingleClass()` class but with the following differences:
- The number of units in the last layer is equal to the number of classes.
- The activation function used in the last layer is softmax.
- The loss function used is binary cross-entropy.

Rest everything is the same as in the single label classification case.

##### Hyperparameter Tuning

![WandB_1_mlp_multi](figures/WandB_1_mlp_multi.png)

![WandB_2_mlp_multi](figures/WandB_2_mlp_multi.png)

![WandB_3_mlp_multi](figures/WandB_3_mlp_multi.png)

The sweep results based on the above plots are:

![Sweep_mlp_multi](figures/Sweep_mlp_multi.png)

Based on the above results, the best set of hyperparameters are:

|Learning Rate|Epochs|Batch Size|Threshold|Optimizer|Hidden Layers|Activation Functions|
|:-----------:|:----:|:--------:|:-------:|:-------:|:------------:|:-------------------:|
|0.001|100|256|0.3|bgd|Four - [64, 32, 8]|[relu, relu, sigmoid]|

##### Evaluation of best model
Based on the best set of hyperparameters, the model has been evaluated on the validation and test sets. The results are as follows:

For validation set:
|Loss|Soft Accuracy|Hard Accuracy|Precision|Recall|F1 Score|
|:---:|:------:|:-------:|:----:|:----:|:------:|
|0.74089|0.9777|0.0|0.33917|0.97581|0.50337|

For test set:
|Loss|Soft Accuracy|Hard Accuracy|Precision|Recall|F1 Score|
|:---:|:------:|:-------:|:----:|:----:|:------:|
|0.75995|0.97445|0.01|0.34305|0.975|0.50753|

### 3: Multi Perceptron Regression

#### Task 1: Data Analysis and Preprocessing
The dataset used here is `HousingData.csv` which contains the information of some houses and their prices based on various features. The prices are the target variable, stored under the column `MEDV` - median value of owner-occupied homes in $1000s. Following is the visualisation of the features of the dataset:

![Housing Data Pair Plot](figures/PairPlot_HousingData.png)

Observations:
- The features `CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `PTRATIO`, `LSTAT` have a roughly exponential relationship with the target variable.
- The feature `RAD` has a roughly linear relationship with the target variable.
- The dataset is not imbalanced.
- Some features have outliers which may affect the performance of the model.
- Features like `CHAS` and `RAD` are categorical in nature, and `PTRATIO` is discrete. Classifying prices based on these features may not be the best idea.
- Some features have a very high correlation with each other - `RAD` and `TAX`, `ZN` and `INDUS`, `LSTAT` and `RM`.
- Features like `RM` and `LSTAT` have a high co-relation with the target variable.

#### Task 2: Model Building from Scratch
The model `MultiLayerPerceptron_Regression()` is implemented in `models/MLP/mlp_reg.py` and is similar to the `MultiLayerPerceptron_SingleClass()` class but with the following differences:
- The number of units in the last layer is 1.
- The activation function used in the last layer is linear.
- The loss function used is mean squared error. 
- Other loss functions like Binary Cross Entropy, Maximum Absolute Error, etc., have also been implemented.
- The metrics used are MSE, RMSE, R2 Score as opposed to accuracy, precision, recall and f1 score in classification.

Rest all remains the same.

#### Task 3: Hyperparameter Tuning using W&B

![WandB_1_mlp_reg](figures/WandB_1_mlp_reg.png)

![WandB_2_mlp_reg](figures/WandB_2_mlp_reg.png)

The sweep results based on the above plots are:

![Sweep_mlp_reg](figures/Sweep_mlp_reg.png)

Based on the above results, the best set of hyperparameters are:

|Learning Rate|Epochs|Batch Size|Optimizer|Loss Function|Hidden Layers|Activation Functions|
|:-----------:|:----:|:--------:|:-------:|:------------:|:------------:|:-------------------:|
|0.01|100|256|sgd|MSE|Three - [16, 8, 1]|[relu, relu, linear]|

The runs can be found under the `data\internals\3` folder with the name `mlp_reg_HPT.csv` (it is a little large to be displayed here).

#### Task 4: Evaluating Model
Using the best set of hyperparameters, the model has been evaluated on the validation and test sets. The results are as follows:

For validation set:

|Loss|MSE|RMSE|R2 Score|
|:---:|:---:|:---:|:------:|
|0.29937 (MSE), 0.23648 (MAE)|0.29937|0.54715|0.74738|

For test set:

|Loss|MSE|RMSE|R2 Score|
|:---:|:---:|:---:|:------:|
|0.13441 (MSE), 0.26309 (MAE)|0.13441|0.36662|0.88217|

#### Task 5: Mean Squared Error v/s Binary Cross Entropy

For MSE: 
![Effect_MSE_mlp_reg_train](figures/effect_MSE_mlp_reg_train.png)

For BCE:
![Effect_BCE_mlp_reg_train](figures/effect_BCE_mlp_reg_train.png)

Observations:
- The model with BCE loss function converges much faster than the model with MSE loss function. However, the loss for MSE is less than that for BCE.
- The faster convergence of the model with BCE is intuitive as well given the nature of the target variable - binary.
- The model with BCE loss function is more prone to overfitting than the model with MSE loss function. 

However, it becomes important to note that the R2 Score is higher for MSE than for BCE, which explains that although the target variable is binary, its distribution does not follow a sigmoidal curve. Thus, BCE fails to do good.

#### Task 6: Analysis 
MSE is higher forthose datapoints which are far away from the mean, thereby, explaining the higher loss. Since the dataset is standardized, the datapoints which are far away from the mean are the outliers. However, the model is not affected much by outliers or even a particular feature due to th standardisation.

#### Task 7: BONUS
The combined regression and classification model is implemented in `models/MLP/mlp_final.py`.

### 4: AutoEncoder

#### Task 1: Autoencoder Implementation from scratch
The autoencoder is implemented in `models/MLP/autoencoder.py` and uses the baseline model class of `MultiLayerPerceptron_Regression()`. The implementation is similar to that of the regression model but with some differences:
- The model is used for dimensionality reduction and not for prediction.
- The number of units in the last layer is the same as the number of features in the dataset.
- The middle-most layer is used for encoding the data and the last layer is used for decoding the data.
- The number of units in the middle-most layer is the same as the number of features desired in the reduced dataset.
- The activation function used in the last layer is sigmoid (because it is a regression model).
- The loss function used is MSE and MAE, and the same metrics are used to evaluate the model.
- The model has functions such as `get_latent()` (returns the latent space representation of the data) and `evaluate()` (evaluates the model on the data).

Rest everything is the same.

#### Task 2: Training an Autoencoder
I trained the autoencoder on the `spotify` dataset (from a1). To find out the best set of hyperparameters, I sweeped over the required hyperparameters using W&B. Following are the results of the same:

![WandB_AutoEncoder](figures/WandB_AutoEncoder.png)

and 

![Sweep_AutoEncoder](figures/Sweep_AutoEncoder.png)

From the above plots, it is evident that the best set of hyperparameters are:

|Learning Rate|Epochs|Batch Size|Optimizer|Loss Function|Hidden Layers|Activation Functions|
|:-----------:|:----:|:--------:|:-------:|:------------:|:------------:|:-------------------:|
|0.001|100|256|sgd|MAE|Three - [17, 11, 17]|[relu, relu, relu]|

#### Task 3: Running KNN on Latent Space

Following is the comparison of the performance of the KNN classifier on the reduced and full dataset:

| Hyperparams | Reduced Dataset A2 | Reduced Dataset A1 | Full Dataset |
|--|--|--|
| k | 15 | 15 | 15 |
| dist_metric | l1 | l1 |
| features | 4 | 4 | 18 |

On the Validation set:

| Metrics | Reduced Dataset A1 | Reduced Dataset A2 | Full Dataset |
|--|--|--|--|
| Accuracy | 16.25 | 15.05 | 36.88 |
| Precision_Macro | 0.1527 | 0.1491 | 0.3579 |
| Precision_Micro | 0.1625 | 0.1506 | 0.3688 |
| Recall_Macro | 0.1489 | 0.1415 | 0.3366 |
| Recall_Micro | 0.1625 | 0.1506 | 0.3688 |
| F1_Macro | 0.1508 | 0.1452 | 0.3469 |
| F1_Micro | 0.1625 | 0.1506 | 0.3688 |
| inference time (seconds) | 40.04 | 27.20 | 96.36 |

On the Test set:

| Metrics | Reduced Dataset A1| Reduced Dataset A2 |
|--|--|--|
| Accuracy | 15.49 | 15.06 |
| Precision_Macro | 0.1485 | 0.1523 |
| Precision_Micro | 0.1549 | 0.1506 |
| Recall_Macro | 0.1427 | 0.1434 |
| Recall_Micro | 0.1549 | 0.1506 |
| F1_Macro | 0.1455 | 0.1447 |      
| F1_Micro | 0.1549 | 0.1506 |
| inference time (seconds) | 38.27 | 27.49 |

Observations:
- The performance of the KNN classifier is better on the reduced dataset as compared to the full dataset. This is expected as the reduced dataset contains only 4 features as compared to 18 in the full dataset.
- The performance of the KNN classifier is comparable on the reduced dataset for A1 and A2.
- The inference time is much lesser for the reduced dataset in A2 as compared to the reduced dataset in A1 - overall, both are much lesser than that of the full dataset.

#### Task 4: Comparing with MLP

By using the best set of hyperparameters, the classification performance of MLP is compared with that of KNN. For the same number of reduced features, the performance of KNN is better than that of MLP.

Following are the results of the same (using MLP):

On the Test set:

| Metrics | Reduced Dataset A2 |
|--|--|
| Accuracy | 13.31 |
| Precision_Macro | 0.0908 | 
| Precision_Micro | 0.1331 | 
| Recall_Macro | 0.1146 | 
| Recall_Micro | 0.1331 | 
| F1_Macro | 0.1014 |      
| F1_Micro | 0.1331 |

Reasons for the relatively poorer performance of MLP as compared to KNN:
- The features are not linearly separable.
- The number of classes is too high as compared to the number of samples and the number of features.

### 5: Link to W&B
Please find experiments here: [link to experiments on W&B](https://wandb.ai/soham-iiith/projects).
