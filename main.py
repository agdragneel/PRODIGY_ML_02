from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import numpy as np

init_data=pd.read_csv("Mall_Customers.csv")

#Remove the customer identifier from datapoint
data = pd.get_dummies(init_data,drop_first=True)
data = data.drop(columns=['CustomerID'])


#Scale the data to range
scaler=StandardScaler()

X_scaled=scaler.fit_transform(data)

# Find ideal number of clusters, Method Used: Elbow Method

'''For a range of values of K, plot the graph of K vs WSS. The elbow turn will give the ideal K value'''

wcss=[]
for i in range(2,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=40)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


plt.plot(range(2, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


n_clusters = 5 #Taken from Graph, optimal elbow bending point

#Apply K Means wiht n_clusters

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

# Store centroid values, and add them as a column
cluster_labels = kmeans.fit_predict(X_scaled)

init_data['Cluster'] = cluster_labels
init_data.to_csv("Clustered_Table.csv",index=False)