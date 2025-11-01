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