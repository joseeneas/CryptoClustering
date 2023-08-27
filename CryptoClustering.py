# %%
#
# Import required libraries and dependencies
#

from   datetime              import datetime;

print('Loading Libraries',datetime.now());

#
# Intentionally ignoring certain waring messages that we know are not relevant to this program, l
# like deprecated functions or changed default parameters.
# This is done to avoid cluttering the output with warnings that are not relevant to this program.
# other condittions with higher seriousness will still be displayed.
#

import warnings
warnings.filterwarnings("ignore")

import pandas                    as pd;
import hvplot.pandas;
from   sklearn.cluster       import KMeans;
from   sklearn.decomposition import PCA;
from   sklearn.preprocessing import StandardScaler;

print('Libraries Loaded ',datetime.now());

# 
# displaying time for loading libraries, just to evaluate the time it takes to load libraries
# program is being tested under two architectures:
# one is a MacBook Pro M2 Ultra with 32GB of RAM.
# the other is a Alienware M18 with 32 GB of RAM.
#

# %%
#
# Data Dicitionary, with variables used in the program
#
# Camel Case variables are used for variables that are used in the program
#
# df_Source_Data        :  DataFrame with the source data, CryptoCurrency data
# crypto_Scaled_Array   :  Array with the scaled data
# crypto_Transformed    :  DataFrame with the transformed data
# coins_Names           :  Array with the coins names
# inertia_1             :  Array with the inertia data
# inertia_2             :  Array with the inertia data
# elbow_Data_1          :  dictionary with the Elbow data 1
# elbow_Data_2          :  dictionary with the Elbow data 2
# elbow_DF_1            :  DataFrame with the Elbow data 1
# elbow_DF_2            :  DataFrame with the Elbow data 2
# clusters_Predicted    :  Array with the predicted clusters
# cluster_PCA_Data      :  DataFrame with the PCA data
# cluster_PCA_DF        :  DataFrame with the PCA data and the predicted clusters
#

# %%
# 
# Load the data into a Pandas DataFrame
#
# The overall asumption is that the data is already clean and ready to be used
#

print('Loading Data',datetime.now())
df_Source_Data = pd.read_csv("Resources/crypto_market_data.csv",index_col="coin_id")

#
# Display sample data
#

print(df_Source_Data.head(10))
print('Data Loaded ',datetime.now())

# %%
#
# Generate summary statistics
#

print('Summary Statistics',datetime.now())
df_Source_Data.describe()

# %%
#
# Plot your data to see what's in your DataFrame
#

print('Plotting Data',datetime.now())
df_Source_Data.hvplot.line(width=1200,height=600,rot=90)

# %% [markdown]
# ---

# %% [markdown]
# ### Prepare the Data

# %%
#
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
# According to the graph above, at least one cryptocurrency is in the thousands, while the rest are in the hundreds.
# This means that the data is skewed and needs to be normalized.
#

print('Normalizing Data',datetime.now())
crypto_Scaled_Array = StandardScaler().fit_transform(df_Source_Data)
print(crypto_Scaled_Array)
print('Data Normalized',datetime.now())

# %%
#
# Create a DataFrame with the scaled data
#

print('Before DataFrame transformation',datetime.now())
crypto_Transformed = pd.DataFrame(crypto_Scaled_Array, columns=['price_change_percentage_24h', 
                                                                'price_change_percentage_7d',
                                                                'price_change_percentage_14d', 
                                                                'price_change_percentage_30d', 
                                                                'price_change_percentage_60d', 
                                                                'price_change_percentage_200d',	
                                                                'price_change_percentage_1y'])
print(crypto_Transformed)

#
# Copy the crypto names from the original data
#

coins_Names                   = list(df_Source_Data.index) #create a list of the coins names
print(coins_Names)

#
# Set the coinid column as index
#

crypto_Transformed['coin_id'] = coins_Names                             #create a new column with the coin names
crypto_Transformed            = crypto_Transformed.set_index('coin_id') #set the coin names as index

#
# Display sample data
#

print('After DataFrame transformation',datetime.now())
print(crypto_Transformed)

# %% [markdown]
# ---

# %% [markdown]
# ### Find the Best Value for k Using the Original Data.

# %%
#
# Create a list with the number of k-values from 1 to 11
#

k = list(range(1,12))
print('Values of k',k)

# %%
#
# Create an empty list to store the inertia values
#

inertia_1 = []

#
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
#

for i in k:
    k_model = KMeans(n_clusters=i, random_state=1,n_init=10,max_iter=1000) #n_init=10,max_iter=1000 provided to avoid depcrecation warning
    k_model.fit(crypto_Transformed)
    inertia_1.append(k_model.inertia_)
print('Values of Inertia ',inertia_1)

# %%
#
# Create a dictionary with the data to plot the Elbow curve
#

print('Creating Elbow Curve',datetime.now())
elbow_Data_1 = {"k": k, "inertia": inertia_1}

#
# Create a DataFrame with the data to plot the Elbow curve
#

elbow_DF_1 = pd.DataFrame(elbow_Data_1)
elbow_DF_1

# %%
#
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
#

print('Plotting Elbow Curve',datetime.now())
elbow_DF_1.hvplot.line(x="k",y="inertia",title= "Elbow Curve",xticks=k,width=1200,height=600)

# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# **Answer:** The best value for 'k' is **4**

# %% [markdown]
# ---

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the Original Data

# %%
#
# Initialize the K-Means model using the best value for k
#

print('Initializing K-Means Model',datetime.now())
model = KMeans(n_clusters=4, random_state=1)
print('K-Means Model Initialized ',datetime.now())

# %%
#
# Fit the K-Means model using the scaled data
#

print('Fitting K-Means Model',datetime.now())
model.fit(crypto_Transformed)
print('K-Means Model Fitted ',datetime.now())

# %%
#
# Predict the clusters to group the cryptocurrencies using the scaled data
#

print('Predicting Clusters',datetime.now())
k4 = model.predict(crypto_Transformed)

#
# Print the resulting array of cluster values.
#

print(k4)
print('Clusters Predicted ',datetime.now())

# %%
#
# Create a copy of the DataFrame
#

print('Creating copy of DataFrame',datetime.now())
clusters_Predicted = crypto_Transformed.copy()
print('Copy of DataFrame Created ',datetime.now())

# %%
#
# Add a new column to the DataFrame with the predicted clusters
#

print('Adding Predicted Clusters',datetime.now())
clusters_Predicted['predicted_cluster'] = k4

# Display the updated data

print(clusters_Predicted.head())
print('Predicted Clusters Added',datetime.now())

# %%
#
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
#

print('Plotting Predicted Clusters',datetime.now())
clusters_Predicted.hvplot.scatter(x='price_change_percentage_24h',y='price_change_percentage_7d',
                                  by='predicted_cluster',hover_cols = 'coin_id',
                                  legend='top_right',width=1200,height=600)

# %% [markdown]
# ---

# %% [markdown]
# ### Optimize Clusters with Principal Component Analysis.

# %%
#
# Create a PCA model instance and set `n_components=3`.
#

print('Creating PCA Model',datetime.now())
pca = PCA(n_components=3)
print('PCA Model Created ',datetime.now())  

# %%
#
# Use the PCA model with `fit_transform` to reduce to three principal components.
#

print('Fitting PCA Model',datetime.now())   
clusters_PCA = pca.fit_transform(clusters_Predicted)
print('PCA Model Fitted ',datetime.now())

#
# View the first five rows of the DataFrame. 
#

clusters_PCA[:5]

# %%
#
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
#

print('Explained Variance',datetime.now())  
pca.explained_variance_ratio_

# %%
#
# Calculate the Total Explained Variance by summing all 3 Explained Variance Ratios
#

print('Total Explained Variance',datetime.now())
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
#
# Create a new DataFrame with the PCA data.
#

print('Creating PCA DataFrame',datetime.now())  
cluster_PCA_df = pd.DataFrame(clusters_PCA,columns = ["PCA1", "PCA2", "PCA3"])
print(cluster_PCA_df)

#
# Copy the crypto names from the original data
#

cluster_PCA_df['coin_id'] = list(clusters_Predicted.index)
print(cluster_PCA_df)

#
# Set the coinid column as index
#

cluster_PCA_df = cluster_PCA_df.set_index('coin_id')

#
# Display sample data
#

print(cluster_PCA_df)
print('PCA DataFrame Created ',datetime.now())

# %% [markdown]
# ---

# %% [markdown]
# ### Find the Best Value for k Using the PCA Data

# %%
#
# Create a list with the number of k-values from 1 to 11
#

k = list(range(1,12))
print('k-values ',k)

# %%
#
# Create an empty list to store the inertia values
#

inertia_2 = []

#
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
#

for i in k:
    k_model = KMeans(n_clusters=i, random_state=1,n_init=10,max_iter=1000)
    k_model.fit(cluster_PCA_df)
    inertia_2.append(k_model.inertia_)
print('Values of Inertia 2 ', inertia_2)

# %%
#
# Create a dictionary with the data to plot the Elbow curve
#

print('Creating Elbow Curve 2',datetime.now())  
elbow_Data_2 = {"k": k, "inertia": inertia_2}

#
# Create a DataFrame with the data to plot the Elbow curve
#

elbow_DF_2 = pd.DataFrame(elbow_Data_2)
print(elbow_DF_2)
print('Elbow Curve 2 Created ',datetime.now())

# %%
#
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
#

print('Plotting Elbow Curve 2',datetime.now())
elbow_DF_2.hvplot.line(x="k", title="Elbow Curve", xticks=k,width=1200,height=600)

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
#
# Initialize the K-Means model using the best value for k
#

print('Initializing K-Means Model 2',datetime.now())    
model = KMeans(n_clusters=4, random_state=1)
print('K-Means Model 2 Initialized ',datetime.now())

# %%
#
# Fit the K-Means model using the PCA data
#

print('Fitting K-Means Model 2',datetime.now())
model.fit(cluster_PCA_df)
print('K-Means Model 2 Fitted ',datetime.now())

# %%
#
# Predict the clusters to group the cryptocurrencies using the PCA data
#

k4 = model.predict(cluster_PCA_df)

#
# Print the resulting array of cluster values.
#

k4

# %%
#
# Create a copy of the DataFrame with the PCA data
#

print('Creating copy of DataFrame with PCA data',datetime.now())    
copy_Cluster_PCA_df= cluster_PCA_df.copy()

#
# Add a new column to the DataFrame with the predicted clusters
#

copy_Cluster_PCA_df['predicted_cluster'] = k4

#
# Display sample data
#

print(copy_Cluster_PCA_df)
print('Copy of DataFrame with PCA data Created ',datetime.now())

# %%
#
# Create a scatter plot using hvPlot by setting 
# `x="PCA1"` and `y="PCA2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
#

print('Plotting Predicted Clusters 2',datetime.now())   
copy_Cluster_PCA_df.hvplot.scatter(x="PCA1",y="PCA2",by ='predicted_cluster',
                                   hover_cols='coin_id',legend='top_right',width=1200,height=600)

# %% [markdown]
# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

# %%
#
# Composite plot to contrast the Elbow curves
#

elbow_DF_1.hvplot.line(x="k", y="inertia", title="Elbow Curve 1", xticks=k) + \
elbow_DF_2.hvplot.line(x="k", y="inertia", title="Elbow Curve 2", xticks=k)

# %%
#
# Composite plot to contrast the clusters
#

clusters_Predicted.hvplot.scatter(x='price_change_percentage_24h',  
                                  y='price_change_percentage_7d', 
                                  by='predicted_cluster', hover_cols = 'coin_id') + \
copy_Cluster_PCA_df.hvplot.scatter(x="PCA1", y="PCA2", by = 'predicted_cluster', hover_cols='coin_id')

# %% [markdown]
# #### Answer the following question: 
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** The impact of using PCA data resulted in tighter grouped clusters, with more entries within cluster 0 and cluster 1 than the original analysis did.


