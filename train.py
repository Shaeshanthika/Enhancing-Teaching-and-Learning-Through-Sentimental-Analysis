import numpy as np
import pandas as pd
import librosa
import os
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract features from audio files
def extract_features(file_path):
    try:
        audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

    return mfccsscaled

# Path to the directory containing the dataset
dataset_dir = "dataset"

# Initialize empty lists to store features and labels
features = []
labels = []

# Loop through each class folder in the dataset directory
for class_label in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_label)
    if os.path.isdir(class_dir):
        # Loop through each audio file in the class folder
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                # Extract features from the audio file
                file_path = os.path.join(class_dir, filename)
                data = extract_features(file_path)
                # If features are extracted successfully, append them to the list along with the label
                if data is not None:
                    features.append(data)
                    labels.append(class_label)

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot pie chart for class distribution
plt.figure(figsize=(8, 8))
class_distribution = pd.Series(y).value_counts()
class_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Class Distribution')
plt.ylabel('')
plt.show()

# Plot bar graph for class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=y, hue=y, palette='viridis', dodge=False)
plt.title('Class Distribution')
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.show()

# Encode class labels to numerical values
class_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
y_encoded = np.array([class_mapping[label] for label in y])

# Plot scatter plot for the first two features
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap='viridis')
plt.title('Scatter Plot of First Two Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class Label')
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the trained model to a file
joblib.dump(nb_classifier, "naive_bayes_model.pkl")
