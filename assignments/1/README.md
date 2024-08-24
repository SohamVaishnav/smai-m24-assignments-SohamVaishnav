# Assignment 1 Report

## Author: Soham Vaishnav
## Roll No.: 2022112002

### 2: K-Nearest Neighbors

#### Task 1: Data Visualisation
The spotify dataset contains 114000 data points with 20 features. The features look as follows:
| Feature Name | Data Type |
|---|---|
| track_id | string |
| artists | string | 
| album_name | string |
| track_name | string |
| popularity | float |
| duration_ms | float |
| explicit | bool |
| danceability | float |
| energy | float |
| key | int |
| loudness | float |
| mode | int |
| speechiness | float |
| acousticness | float |
| instrumentalness | float |
| liveliness | float |
| valence | float |
| tempo | float |
| time_signature | int |
| track_genre | string |

Since the task is to deploy KNN for predicting music genre, the last feature becomes the label set. For the other columns, the string ones can be either removed or encoded into some numeric form for them to be useable by the model. \\
For visualisation, I have plotted the numeric features. The individual figures can be seen in `./figure`. For the combined plots, refer to the following figures:
To add a figure in the README.md file, you can use the following markdown syntax:


![Features Before and After preprocessing](./figures/All_Features.png)

For some features like acousticness, instrumentalness, tempo and valence, after looking at the distribution, it felt better to represent them in the following manner as well:

| ![Acousticness](./figures/Acousticness_aliter.png) | ![Instrumentalness](./figures/Instrumentalness_aliter.png) | 
|---|---|

| ![Tempo](./figures/Tempo_aliter.png) | ![Valence](./figures/Valence_aliter.png) |
|---|---|

Here, we can clearly observe the effects of normalisation on the data (covered in the next section). \\
For the labels, I visualised their distribution using a histogram so as to give an idea about how to split the data into train, validation and test sets.

![Features Before and After preprocessing](./figures/genre_distribution.png)

Initially, all the classes were equally distributed across the datapoints and thus it did not make sense to perform stratified sampling. But if we observe the data after preprocessing, it is clear that stratified sampling is required (covered in the next section). \\

Since, we had to visualise the correlation between various features, I felt it most appropriate to generate a pair plot between features over the labels so as to give an idea of how they all are related.

![Features Before and After preprocessing](./figures/PairPlot_BP.png)






