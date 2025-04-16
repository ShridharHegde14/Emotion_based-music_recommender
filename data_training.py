import os  
import numpy as np 
import joblib
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# ---------------------
# Load and Process Data
# ---------------------
is_init = False
X = []
y_labels = []

label_dict = {}
label_counter = 0

for i in os.listdir("collected_data"):
    if i.endswith(".npy"):
        path = os.path.join("collected_data", i)
        data = np.load(path)
        
        # Get base label from filename: 'happy_1.npy' → 'happy'
        base_label = i.split('.')[0].split('_')[0]

        # Assign a unique index for each base label
        if base_label not in label_dict:
            label_dict[base_label] = label_counter
            label_counter += 1

        label_index = label_dict[base_label]
        X.append(data)
        y_labels.extend([label_index] * data.shape[0])

# Convert to numpy arrays
X = np.vstack(X)
y = np.array(y_labels)

# ---------------------
# Encode labels
# ---------------------
y = to_categorical(y)

# ---------------------
# Shuffle data
# ---------------------
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# ---------------------
# Apply PCA
# ---------------------
print("Original shape:", X.shape)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

print("PCA-reduced shape:", X_pca.shape)

# Save PCA model
joblib.dump(pca, "pca.pkl")

# ---------------------
# Build the Model
# ---------------------
ip = Input(shape=(X_pca.shape[1],))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])

# ---------------------
# Train the Model
# ---------------------
model.fit(X_pca, y, epochs=50)

# ---------------------
# Save Model + Clean Labels
# ---------------------
model.save("model.h5")

# Save labels in order of their index
reverse_label_dict = {v: k for k, v in label_dict.items()}
labels = [reverse_label_dict[i] for i in range(len(reverse_label_dict))]
np.save("labels.npy", np.array(labels))

print("✅ Model, labels, and PCA saved successfully!")
