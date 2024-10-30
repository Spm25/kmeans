# Step 1: Load the data
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

df = pd.read_csv("Movies_Dataset_2.csv")

# Hiển thị 5 dòng đầu tiên của DataFrame
df.head()

# Step 2: Explore the data
df.info()

# Step 3: Data preprocessing
documents = df['overview'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)

# Chạy Silhouette Score cho nhiều giá trị của k
silhouette_scores = []
k_values = range(2, 30)  # thử nghiệm với k từ 2 đến 10

for k in k_values:
    # model = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, batch_size=100)
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    model.fit(features)
    silhouette_avg = silhouette_score(features, model.labels_)
    silhouette_scores.append(silhouette_avg)
    print(f'K={k}, Silhouette Score: {silhouette_avg:.3f}')

# Vẽ biểu đồ Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.grid()
plt.show()