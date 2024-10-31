import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import os

# Lấy đường dẫn file đầu vào và đầu ra từ tham số dòng lệnh
input_file = sys.argv[1]
output_folder = sys.argv[2]
k = int(sys.argv[3])  # số cụm k

# Step 1: Load the data
df = pd.read_csv(input_file)

# Hiển thị 5 dòng đầu tiên của DataFrame
df.head()

# Step 2: Explore the data
df.info()

# Step 3: Data preprocessing
documents = df['overview'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)

# Sử dụng K-Means
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(features)

# Gán nhãn cụm cho từng phim
df['cluster'] = model.labels_


# Xuất kết quả ra các file CSV, mỗi file đại diện cho một cụm
clusters = df.groupby('cluster')

for cluster in clusters.groups:
    # Tạo đường dẫn lưu file cho từng cụm
    output_file = os.path.join(output_folder, f'cluster_{cluster}.csv')
    data = clusters.get_group(cluster)[['title', 'overview']]
    data.to_csv(output_file, index_label='id', encoding='utf-8')

# In ra các tâm cụm
print("Cluster centroids:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(k):
    print(f"Cluster {i}:")
    for j in order_centroids[i, :10]:
        print(f' {terms[j]}')
    print('------------')
