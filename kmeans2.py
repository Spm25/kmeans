# Step 1: Load the data
import pandas as pd
df = pd.read_csv("Movies_Dataset_1.csv")

# Hiển thị 5 dòng đầu tiên của DataFrame
df.head()

# Step 2: Explore the data
df.info()

# Step 3: Data preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score  # Thêm import cho Silhouette Score

documents = df['overview'].values.astype("U")

# Chuyển đổi văn bản thành vector đặc trưng với TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)

# Số lượng cụm
k = 5

# Sử dụng Mini-Batch K-Means
model = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, batch_size=100)
model.fit(features)

# Gán nhãn cụm cho từng phim
df['cluster'] = model.labels_

# Hiển thị 5 dòng đầu tiên của DataFrame với cột 'cluster'
df.head()

# Tính toán và in ra Silhouette Score
silhouette_avg = silhouette_score(features, model.labels_)  # Tính Silhouette Score
print(f'Silhouette Score: {silhouette_avg:.3f}')  # In ra Silhouette Score

# Xuất kết quả ra file CSV
clusters = df.groupby('cluster')

for cluster in clusters.groups:
    f = open('cluster' + str(cluster) + '.csv', 'w', encoding='utf-8')  # thêm encoding='utf-8'
    data = clusters.get_group(cluster)[['title', 'overview']]  # lấy cột title và overview
    f.write(data.to_csv(index_label='id'))  # thiết lập chỉ mục là id
    f.close()

# In ra các tâm cụm
print("Cluster centroids: \n")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(k):
    print("Cluster %d:" % i)
    for j in order_centroids[i, :10]:  # in ra 10 từ đặc trưng của mỗi cụm
        print(' %s' % terms[j])
    print('------------')
