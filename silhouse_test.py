import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# Lấy đường dẫn file đầu vào từ tham số dòng lệnh
input_file = sys.argv[1]
k_input = int(sys.argv[2])

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

# Chạy Silhouette Score cho nhiều giá trị của k
silhouette_scores = []
k_values = range(2, k_input)  # thử nghiệm với k từ 2 đến 30

for k in k_values:
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
