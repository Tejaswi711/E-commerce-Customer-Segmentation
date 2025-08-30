import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("Ecommerce_sample.csv")

# Ensure proper datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Remove missing CustomerID rows (important for RFM)
df = df.dropna(subset=["CustomerID"])

# Calculate Total Amount per transaction
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# ------------------------------
# RFM Feature Engineering
# ------------------------------
# Define snapshot date as 1 day after the latest purchase
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
    "InvoiceNo": "nunique",  # Frequency
    "TotalAmount": "sum"     # Monetary
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# ------------------------------
# Data Preprocessing
# ------------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# ------------------------------
# KMeans Clustering
# ------------------------------
# Choose clusters dynamically (min between 4 and number of customers)
n_clusters = min(4, len(rfm))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# ------------------------------
# Cluster Summary
# ------------------------------
cluster_summary = rfm.groupby("Cluster").mean().round(2)
print("Cluster Summary:\n")
print(cluster_summary)

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster", palette="tab10", s=100)
plt.title("Customer Segmentation (RFM Clusters)")
plt.xlabel("Recency (days)")
plt.ylabel("Monetary Value")
plt.legend(title="Cluster")
plt.show()
