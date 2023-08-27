# %%
#
# Import required libraries and dependencies
#
import warnings
warnings.filterwarnings("ignore")

from   datetime              import datetime

print('Loading Libraries',datetime.now());

import pandas                    as pd;
import hvplot.pandas;
from   sklearn.cluster       import KMeans;
from   sklearn.decomposition import PCA;
from   sklearn.preprocessing import StandardScaler;

print('Libraries Loaded ',datetime.now());

# displaying time for loading libraries, just to evaluate the time it takes to load libraries
# program is being tested under two architectures:
# one is a MacBook Pro M2 Ultra with 32GB of RAM.
# the other is a Alienware M18 with 32 GB of RAM.

# %%
#
# Data Dicitionary, with variables used in the program
#
# dfSourceData        :  DataFrame with the source data, CryptoCurrency data
# cryptoScaledArray   :  Array with the scaled data
# cryptoTransformed   :  DataFrame with the transformed data
# coinsNames          :  Array with the coins names
# inertia1            :  Array with the inertia data
# inertia2            :  Array with the inertia data
# elbowData1          :  dictionary with the Elbow data 1
# elbowData2          :  dictionary with the Elbow data 2
# elbowDF1            :  DataFrame with the Elbow data 1
# elbowDF2            :  DataFrame with the Elbow data 2
# clustersPredicted   :  Array with the predicted clusters
# clusterPCAData      :  DataFrame with the PCA data
# clusterPCADF        :  DataFrame with the PCA data and the predicted clusters

# %%
# 
# Load the data into a Pandas DataFrame
#

print('Loading Data',datetime.now())
dfSourceData = pd.read_csv("Resources/crypto_market_data.csv",index_col="coin_id")

#
# Display sample data
#

print(dfSourceData.head(10))
print('Data Loaded ',datetime.now())

# %%
# Generate summary statistics
print('Summary Statistics',datetime.now())
dfSourceData.describe()

# %%
#
# Plot your data to see what's in your DataFrame
#
print('Plotting Data',datetime.now())
dfSourceData.hvplot.line(width=1200,height=600,rot=90)

# %% [markdown]
# ### Prepare the Data

# %%
#
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
# According to the graph above, at least one cryptocurrency is in the thousands, while the rest are in the hundreds.
# This means that the data is skewed and needs to be normalized.
# 
print('Normalizing Data',datetime.now())
cryptoScaledArray = StandardScaler().fit_transform(dfSourceData)
print(cryptoScaledArray)
print('Data Normalized',datetime.now())

# %%
#
# Create a DataFrame with the scaled data
#
print('Before DataFrame transformation',datetime.now())
cryptoTransformed = pd.DataFrame(cryptoScaledArray, columns=['price_change_percentage_24h', 
                                                             'price_change_percentage_7d',
                                                             'price_change_percentage_14d', 
                                                             'price_change_percentage_30d', 
                                                             'price_change_percentage_60d', 
                                                             'price_change_percentage_200d',	
                                                             'price_change_percentage_1y'])
print(cryptoTransformed)
#
# Copy the crypto names from the original data
#
coinsNames                   = list(dfSourceData.index) #create a list of the coins names
print(coinsNames)
#
# Set the coinid column as index
#
cryptoTransformed['coin_id'] = coinsNames                             #create a new column with the coin names
cryptoTransformed            = cryptoTransformed.set_index('coin_id') #set the coin names as index
#
# Display sample data
#
print('After DataFrame transformation',datetime.now())
print(cryptoTransformed)


# %% [markdown]
# ### Find the Best Value for k Using the Original Data.

# %%
# Create a list with the number of k-values from 1 to 11
k = list(range(1,12))
print('Values of k',k)

# %%
#
# Create an empty list to store the inertia values
#
inertia1 = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=1,n_init=10,max_iter=1000) #n_init=10,max_iter=1000 provided to avoid depcrecation warning
    k_model.fit(cryptoTransformed)
    inertia1.append(k_model.inertia_)
print('Values of Inertia ',inertia1)

# %%
# Create a dictionary with the data to plot the Elbow curve
print('Creating Elbow Curve',datetime.now())
elbowData1 = {"k": k, "inertia": inertia1}


# Create a DataFrame with the data to plot the Elbow curve
elbowDF1 = pd.DataFrame(elbowData1)
elbowDF1

# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
print('Plotting Elbow Curve',datetime.now())
elbowDF1.hvplot.line(
    x="k",
    y="inertia",
    title= "Elbow Curve",
    xticks=k
)

# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# **Answer:** The best value for 'k' is **4**

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the Original Data

# %%
# Initialize the K-Means model using the best value for k
print('Initializing K-Means Model',datetime.now())
model = KMeans(n_clusters=4, random_state=1)
print('K-Means Model Initialized ',datetime.now())

# %%
# Fit the K-Means model using the scaled data
print('Fitting K-Means Model',datetime.now())
model.fit(cryptoTransformed)
print('K-Means Model Fitted ',datetime.now())

# %%
# Predict the clusters to group the cryptocurrencies using the scaled data
k4 = model.predict(cryptoTransformed)

# Print the resulting array of cluster values.
k4


# %%
# Create a copy of the DataFrame
clustersPredicted = cryptoTransformed.copy()

# %%
# Add a new column to the DataFrame with the predicted clusters
clustersPredicted['predicted_cluster'] = k4

# Display sample data
clustersPredicted.head()

# %%
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
clustersPredicted.hvplot.scatter(x='price_change_percentage_24h',y='price_change_percentage_7d',by='predicted_cluster',hover_cols = 'coin_id'
)

# %% [markdown]
# ### Optimize Clusters with Principal Component Analysis.

# %%
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)


# %%
#
# Use the PCA model with `fit_transform` to reduce to three principal components.
#
clustersPCA = pca.fit_transform(clustersPredicted)
#
# View the first five rows of the DataFrame. 
#
clustersPCA[:5]

# %%
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_

# %%
#calculate the Total Explained Variance by summing all 3 Explained Variance Ratios
sum(pca.explained_variance_ratio_)

# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** 0.34871677 + 0.31363391 + 0.22627118 = 0.88862186
# 
# **Answer may change depending on re-execution of the whole code**

# %%
# Create a new DataFrame with the PCA data.
clusterPCAdf = pd.DataFrame(
    clustersPCA,
    columns = ["PCA1", "PCA2", "PCA3"]
)
clusterPCAdf

# Copy the crypto names from the original data
clusterPCAdf['coin_id'] = list(clustersPredicted.index)
clusterPCAdf

# Set the coinid column as index
clusterPCAdf = clusterPCAdf.set_index('coin_id')

# Display sample data
clusterPCAdf


# %% [markdown]
# ### Find the Best Value for k Using the PCA Data

# %%
# Create a list with the number of k-values from 1 to 11
k = list(range(1,12))
k

# %%
# Create an empty list to store the inertia values
inertia2 = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=1,n_init=10,max_iter=1000)
    k_model.fit(clusterPCAdf)
    inertia2.append(k_model.inertia_)
print(inertia2)

# %%
# Create a dictionary with the data to plot the Elbow curve
elbowData2 = {"k": k, "inertia": inertia2}
# Create a DataFrame with the data to plot the Elbow curve
elbowDF2 = pd.DataFrame(elbowData2)
elbowDF2

# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbowDF2.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

# %% [markdown]
# #### Answer the following questions: 
# 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:** The best k-value is  `k=4` when using PCA data
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** No, it is the same `k` value as found using the original data

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# %%
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=1)

# %%
# Fit the K-Means model using the PCA data
model.fit(clusterPCAdf)

# %%
# Predict the clusters to group the cryptocurrencies using the PCA data
k4 = model.predict(clusterPCAdf)
# Print the resulting array of cluster values.
k4

# %%
# Create a copy of the DataFrame with the PCA data
copyClusterPCAdf= clusterPCAdf.copy()

# Add a new column to the DataFrame with the predicted clusters
copyClusterPCAdf['predicted_cluster'] = k4

# Display sample data
copyClusterPCAdf

# %%
# Create a scatter plot using hvPlot by setting 
# `x="PCA1"` and `y="PCA2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
copyClusterPCAdf.hvplot.scatter(
    x="PCA1",
    y="PCA2",
    by = 'predicted_cluster',
    hover_cols='coin_id'
)

# %% [markdown]
# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

# %%
# Composite plot to contrast the Elbow curves
elbowDF1.hvplot.line(x="k", y="inertia", title="Elbow Curve 1", xticks=k) + \
elbowDF2.hvplot.line(x="k", y="inertia", title="Elbow Curve 2", xticks=k)

# %%
# Composite plot to contrast the clusters
clustersPredicted.hvplot.scatter( x='price_change_percentage_24h',  
                                  y='price_change_percentage_7d', 
                                  by='predicted_cluster', hover_cols = 'coin_id') + \
copyClusterPCAdf.hvplot.scatter(x="PCA1", y="PCA2", by = 'predicted_cluster', hover_cols='coin_id')

# %% [markdown]
# #### Answer the following question: 
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** The impact of using PCA data resulted in tighter grouped clusters, with more entries within cluster 0 and cluster 1 than the original analysis did.


