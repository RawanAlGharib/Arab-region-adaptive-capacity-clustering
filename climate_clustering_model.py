##Phase 1: Data Acquisition & Pre-Processing (The Economic Drivers)


import wbgapi as wb
import pandas as pd

# 1. Define the complete 22 Arab League member states
arab_states = [
    'DZA', 'BHR', 'COM', 'DJI', 'EGY', 'IRQ', 'JOR', 'KWT', 'LBN', 
    'LBY', 'MRT', 'MAR', 'OMN', 'PSE', 'QAT', 'SAU', 'SOM', 'SDN', 
    'SYR', 'TUN', 'ARE', 'YEM'
]

# 2. Define the exact World Bank Indicator codes for our drivers
# These map directly to the economic, human capital, and infrastructure pillars
indicators = {
    'NY.GDP.PCAP.KD': 'GDP_per_Capita',         # GDP per capita (constant 2015 US$)
    'SL.TLF.CACT.FE.ZS': 'Female_LFP_%',        # Female Labor Force Participation
    'NV.AGR.TOTL.ZS': 'Agri_GDP_%',             # Agriculture, forestry, fishing (% of GDP)
    'IT.NET.USER.ZS': 'Internet_%'              # Individuals using the Internet (%)
}

# 3. Fetch the data for a specific recent year (e.g., 2022 to minimize missing data)
print("Fetching data from the World Bank API...")
raw_data = wb.data.DataFrame(indicators.keys(), arab_states, time=2022)

# 4. Clean and reshape the matrix for the K-Means algorithm
# Reset the index so 'economy' becomes a regular column
df = raw_data.reset_index()

# Rename the columns from the API codes to our clean headers
df = df.rename(columns=indicators)
df = df.rename(columns={'economy': 'Country_Code'})

# 5. Handle missing values (Crucial for distance-based math like K-Means)
# We will drop any country that is missing data across these core indicators
df_clean = df.dropna().reset_index(drop=True)

print("\nData successfully extracted and cleaned!")
print("-" * 40)
print(df_clean.head())

# 6. Export to CSV for safekeeping
df_clean.to_csv('arab_region_adaptive_capacity_data.csv', index=False)




##Phase 2: Dimensionality Reduction (The Math)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the cleaned dataset from Phase 1
df = pd.read_csv('arab_region_adaptive_capacity_data.csv')

# 2. Separate the text (Country Codes) from the numbers
# Algorithms cannot do math on letters, so we isolate the numeric columns
features = ['GDP_per_Capita', 'Female_LFP_%', 'Agri_GDP_%', 'Internet_%']
x = df.loc[:, features].values
countries = df['Country_Code'].values

# 3. Standardize the data (Mean=0, Variance=1)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 4. Apply PCA to compress the data down to 2 Principal Components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_scaled)

# 5. Rebuild a clean DataFrame with our new mathematical coordinates
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Country_Code'] = countries # Add the country names back in

# Reorder so Country Code is the first column
pca_df = pca_df[['Country_Code', 'PC1', 'PC2']]

print("PCA Transformation Complete!")
print(f"Variance explained by these 2 components: {sum(pca.explained_variance_ratio_) * 100:.2f}%")
print("-" * 40)
print(pca_df.head())


##Phase 3: The K-Means Engine & The Elbow Method
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Extract our two Principal Components for clustering
X_cluster = pca_df[['PC1', 'PC2']].values

# 2. Calculate WCSS for k=1 through k=10
wcss = []
for i in range(1, 11):
    # Initialize K-Means with 'k-means++' to ensure smart starting centroids
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_cluster)
    
    # inertia_ is the scikit-learn attribute for WCSS
    wcss.append(kmeans.inertia_)

# 3. Plot the Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method: Optimal Clusters for Adaptive Capacity')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

##optimal k=3



# 1. Initialize and fit the final K-Means model with optimal k=3
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)

# 2. Assign each country to its mathematical cluster
pca_df['Cluster_ID'] = kmeans_final.fit_predict(X_cluster)

# Add the cluster assignments back to our ORIGINAL dataframe for policy analysis
df['Cluster_ID'] = pca_df['Cluster_ID'] 

# 3. Visualize the 3 distinct clusters
plt.figure(figsize=(12, 8))

# Plot each cluster with a different color
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(optimal_k):
    cluster_points = pca_df[pca_df['Cluster_ID'] == i]
    plt.scatter(cluster_points['PC1'], cluster_points['PC2'], 
                s=100, c=colors[i], label=f'Cluster {i}')

# Add the Country Codes as text labels next to the dots
for i, txt in enumerate(pca_df['Country_Code']):
    plt.annotate(txt, (pca_df['PC1'][i] + 0.1, pca_df['PC2'][i]), fontsize=10)

# Mark the mathematical center (Centroid) of each group
centroids = kmeans_final.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title('Arab Region Adaptive Capacity: K-Means Clusters (k=3)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Group the original data by our new K-Means clusters and calculate the mean
cluster_analysis = df.groupby('Cluster_ID')[features].mean().round(2)

# Count how many countries are in each cluster
cluster_analysis['Country_Count'] = df.groupby('Cluster_ID')['Country_Code'].count()

print("\n🌍 Centroid Analysis: The Economic Profile of Each Cluster")
print("-" * 60)
print(cluster_analysis)

# Print out the final list of who is in which group
print("\nFinal Cluster Assignments:")
print("-" * 40)
print(df[['Country_Code', 'Cluster_ID']].sort_values(by='Cluster_ID').to_string(index=False))

# Export the final, labeled dataset
df.to_csv('arab_region_clusters_final.csv', index=False)