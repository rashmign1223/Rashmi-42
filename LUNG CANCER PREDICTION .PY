import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
# Step 1: Load Dataset (TAB-separated file)
data = pd.read_csv("lung_cancer_data.csv", sep="\t")

# Fix column names (remove spaces)
data.columns = data.columns.str.strip()

# Convert GENDER
data["GENDER"] = data["GENDER"].map({"M": 1, "F": 0})

# Convert YES/NO to 1/0
data["LUNG_CANCER"] = data["LUNG_CANCER"].map({"YES": 1, "NO": 0})

# Convert entire dataset to numeric where possible
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove empty rows (if any)
data = data.dropna()

# Separate features & target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Neural Network
model = Sequential([
    Dense(32, activation="relu", input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()
