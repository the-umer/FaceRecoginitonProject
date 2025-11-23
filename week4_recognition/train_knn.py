import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

embed_dir = "../week3_embeddings/embeddings"

X = []
y = []

# Load all embedding files
for file in os.listdir(embed_dir):
    if file.endswith(".npy"):
        name = file.replace(".npy", "")
        embeddings = np.load(os.path.join(embed_dir, file))

        for emb in embeddings:
            X.append(emb)
            y.append(name)

X = np.array(X)
y = np.array(y)

print("Training data shape:", X.shape)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X, y)

# Save model
joblib.dump(knn, "knn_model.pkl")

print("✅ KNN model saved as knn_model.pkl")
