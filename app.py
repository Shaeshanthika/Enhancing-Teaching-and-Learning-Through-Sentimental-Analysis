from flask import Flask, flash, request, redirect, url_for, render_template
import numpy as np
import os
import librosa
import joblib

# Configuring Flask
app = Flask(__name__)

# Function to extract features from audio files
def extract_features(file_path):
    try:
        audio_data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None

    return mfccsscaled
def count_classes(predictions):
    class_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for _, predicted_class in predictions:
        class_counts[predicted_class] += 1
    return class_counts

# Load the trained Naive Bayes classifier
nb_classifier = joblib.load("naive_bayes_model.pkl")

@app.route('/')
def home():
    bgimg='background.jpg'
    bgimgpath = url_for('static', filename=bgimg)
    return render_template('sentiment.html', bgimgpath=bgimgpath)

@app.route('/reviewform')
def reviewform():
    bgimg='background.jpg'
    bgimgpath = url_for('static', filename=bgimg)
    return render_template('reviewform.html', bgimgpath=bgimgpath)

@app.route('/submitreport', methods=['POST'])
def submit_report():
    if request.method == 'POST':
        # Extracting form data
        staff_name = request.form['StaffName']
        content_relevance = request.form['ContentRelevance']
        engagement_level = request.form['EngagementLevel']
        clarity_of_instruction = request.form['ClarityofInstruction']
        interaction_and_communication = request.form['InteractionandCommunication']
        feedback_on_assignments = request.form['FeedbackonAssignments']
        availability_of_resources = request.form['AvailabilityofResources']
        suggestions_for_improvement = request.form['SuggestionsforImprovement']

        # Creating directory for staff if it doesn't exist
        staff_dir = os.path.join('staff', staff_name)
        if not os.path.exists(staff_dir):
            os.makedirs(staff_dir)

        # Saving form data to a text file inside 'reviews' folder
        reviews_dir = 'reviews'
        if not os.path.exists(reviews_dir):
            os.makedirs(reviews_dir)
        review_file_path = os.path.join(reviews_dir, f'{staff_name}.txt')
        with open(review_file_path, 'w') as file:
            file.write(f"Staff Name: {staff_name}\n")
            file.write(f"Content Relevance: {content_relevance}\n")
            file.write(f"Engagement Level: {engagement_level}\n")
            file.write(f"Clarity of Instruction: {clarity_of_instruction}\n")
            file.write(f"Interaction and Communication: {interaction_and_communication}\n")
            file.write(f"Feedback on Assignments: {feedback_on_assignments}\n")
            file.write(f"Availability of Resources: {availability_of_resources}\n")
            file.write(f"Suggestions for Improvement: {suggestions_for_improvement}\n")

        # Saving uploaded files
        uploaded_files = request.files.getlist('Uploadfeedbacks')
        for file in uploaded_files:
            if file.filename != '':
                if file.filename.endswith('.wav'):
                    file.save(os.path.join(staff_dir, file.filename))

        bgimg='background.jpg'
        bgimgpath = url_for('static', filename=bgimg)
        return render_template('submitreport.html', staff_name=staff_name, content_relevance=content_relevance, engagement_level=engagement_level, clarity_of_instruction=clarity_of_instruction, interaction_and_communication=interaction_and_communication, feedback_on_assignments=feedback_on_assignments, availability_of_resources=availability_of_resources, suggestions_for_improvement=suggestions_for_improvement, bgimgpath=bgimgpath)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        input_name = request.form['Name']  # Replace this with the name you want to predict

        # Path to the directory containing the audio files to predict
        predict_dir = "staff/" + input_name

        # Initialize lists to store predictions and sentiment counts
        predictions = []
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}


        # Loop through each WAV file in the predict folder
        for filename in os.listdir(predict_dir):
            if filename.endswith(".wav"):
                # Extract features from the audio file
                file_path = os.path.join(predict_dir, filename)
                data = extract_features(file_path)
                # If features are extracted successfully, make a prediction
                if data is not None:
                    prediction = nb_classifier.predict([data])[0]
                    predictions.append((filename, prediction))
                    # Update sentiment counts
                    sentiment_counts[prediction] += 1
        class_counts = count_classes(predictions)
        # Print the predictions
        for filename, prediction in predictions:
            print(f"File: {filename}, Predicted Class: {prediction}")
        bgimg='background.jpg'
        bgimgpath = url_for('static', filename=bgimg)
        return render_template('result.html', Name=input_name, r=predictions, sentiment_counts=sentiment_counts, class_counts=class_counts, bgimgpath=bgimgpath)

if __name__ == '__main__':
    app.run(debug=True)
