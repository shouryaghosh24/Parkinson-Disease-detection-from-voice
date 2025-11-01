import os
import numpy as np
import librosa
import joblib  # ✅ added to save scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ✅ Dataset path
data_dir = r"C:\Users\KIIT\Documents\project_ind4.0\data\kaggle_voice"

# ---------------- Feature Extraction ----------------
def extract_features(file_path):
    """Extracts MFCC features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# ---------------- Load Data ----------------
def load_data():
    """Loads audio data and extracts features."""
    X, y = [], []
    print("📂 Loading dataset...")

    for label in ['healthy', 'parkinson']:
        folder_path = os.path.join(data_dir, label)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"✅ Found {len(files)} files in '{label}' folder.")

        for i, file in enumerate(files):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(0 if label == 'healthy' else 1)

            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(files)} files in '{label}'")

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Total samples loaded: {len(X)}")
    return X, y

# ---------------- Data Preparation ----------------
X, y = load_data()

if len(X) == 0:
    print("❌ No data found! Please check dataset path and folder names.")
    exit()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Save the fitted scaler for Streamlit web app
joblib.dump(scaler, "scaler.pkl")
print("💾 Scaler saved as 'scaler.pkl'")

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# ---------------- Model Architecture ----------------
model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------- Training ----------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("\n🚀 Training model...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ---------------- Evaluation ----------------
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n📊 Model Performance Summary:")
print(f"✅ Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"✅ Testing Accuracy: {test_accuracy * 100:.2f}%")

# ---------------- Save Model ----------------
accuracy_percent = int(test_accuracy * 100)
keras_filename = f"parkinson_model_{accuracy_percent}.keras"
h5_filename = f"parkinson_model_{accuracy_percent}.h5"

model.save(keras_filename)
model.save(h5_filename)

print(f"✅ Model saved successfully as '{keras_filename}' and '{h5_filename}'")
print("🎯 Training and saving complete!")

