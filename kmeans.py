import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec
import spacy

# Khởi tạo mô hình spaCy để nhận diện thực thể
nlp = spacy.load("en_core_web_sm")

# Lấy đường dẫn file đầu vào và đầu ra từ tham số dòng lệnh
input_file = sys.argv[1]
output_folder = sys.argv[2]
k = int(sys.argv[3])  # số cụm k

# Step 1: Load the data
df = pd.read_csv(input_file)

# Hiển thị 5 dòng đầu tiên của DataFrame
print(df.head())

# Step 2: Explore the data
print(df.info())

# Step 3: Entity Recognition (Nhận diện thực thể)
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:  # Nhận diện người, địa điểm, tổ chức
            entities.append(ent.text)
    return " ".join(entities)

# Tạo cột mới chứa các thực thể được nhận diện
df["entities"] = df["overview"].apply(extract_entities)

# Step 4: Convert text data to vectors using Word2Vec
# Tạo tập dữ liệu cho Word2Vec
sentences = [text.split() for text in df['overview'].values.astype("U")]

# Khởi tạo mô hình Word2Vec
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Hàm để lấy trung bình vector Word2Vec cho mỗi văn bản
def vectorize_text(text):
    words = text.split()
    vector = sum(w2v_model.wv[word] for word in words if word in w2v_model.wv)
    return vector / len(words) if words else None

# Áp dụng vectorize_text để chuyển đổi văn bản thành vector
df['w2v_vector'] = df['overview'].apply(vectorize_text)

# Bỏ các dòng có giá trị None
df.dropna(subset=['w2v_vector'], inplace=True)

# Step 5: TF-IDF Vectorization
documents = df['overview'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_features = vectorizer.fit_transform(documents)

# Kết hợp TF-IDF và Word2Vec bằng cách tạo thành một DataFrame mới
from scipy.sparse import hstack
import numpy as np

w2v_features = np.vstack(df['w2v_vector'].values)
features = hstack([tfidf_features, w2v_features])

# Step 6: Mini-Batch K-Means Clustering
model = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1, batch_size=100)
model.fit(features)

# Gán nhãn cụm cho từng phim
df['cluster'] = model.labels_

# Tính toán và in ra Silhouette Score
silhouette_avg = silhouette_score(features, model.labels_)
print(f'Silhouette Score: {silhouette_avg:.3f}')

# Xuất kết quả ra các file CSV, mỗi file đại diện cho một cụm
clusters = df.groupby('cluster')
for cluster in clusters.groups:
    output_file = os.path.join(output_folder, f'cluster_{cluster}.csv')
    data = clusters.get_group(cluster)[['title', 'overview', 'entities']]
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
