# Step 1: Importing Essential Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Step 2: Loading the CSV file
from google.colab import files
uploaded = files.upload()
csv_filename = list(uploaded.keys())[0]

# Read the CSV file
df = pd.read_csv(csv_filename)
print("Dataset Preview:\n", df.head())

# Step 3: Handling Categorical and Missing Data
# Encode categorical columns
categorical_columns = ['sex', 'localization']
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Fill missing values (e.g., age) with the mean
if 'age' in df.columns:
    df['age'].fillna(df['age'].mean(), inplace=True)

# Encode the target variable (dx)
df['dx'] = LabelEncoder().fit_transform(df['dx'])
print(f"Target Classes: {df['dx'].unique()}")

# Step 4: Feature Selection and Scaling
# Drop irrelevant columns
columns_to_drop = ['image_id', 'dx_type']  # Drop non-numeric and non-relevant columns
X = df.drop(columns=columns_to_drop + ['dx'])  # Keep only feature columns
y = to_categorical(df['dx'], num_classes=df['dx'].nunique())

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Fill any remaining missing values
X.fillna(0, inplace=True)

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Building
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 6: Training the Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=16)

# Plotting Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 7: Testing the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Classification Report:\n")
print(classification_report(y_true_classes, y_pred_classes))

