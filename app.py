from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# NLTK setup
# --------------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Keep negation words
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

# --------------------------------------------------
# Load models and tools
# --------------------------------------------------
log_reg = joblib.load("models/logistic_regression_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# --------------------------------------------------
# Flask app
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# Preprocessing (matches training)
# --------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(tokens)

# --------------------------------------------------
# Optional post-processing for negation safety
# --------------------------------------------------
def post_process(text, emotion):
    text = text.lower()

    # Example: convert "not happy" predicted as positive → sadness
    if "not happy" in text and emotion in ["happiness", "love", "fun"]:
        return "sadness"
    if "not love" in text and emotion in ["love", "fun"]:
        return "anger"
    # Add more rules if needed
    return emotion

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
def predict_emotion(text, model_name="logreg"):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    if model_name == "svm":
        pred = svm_model.predict(vectorized_text)[0]
    else:  # default logreg
        pred = log_reg.predict(vectorized_text)[0]

    emotion = label_encoder.inverse_transform([pred])[0]
    emotion = post_process(text, emotion)
    return emotion

# --------------------------------------------------
# Web UI route
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("text", "")
        model_name = request.form.get("model", "logreg")
        emotion = predict_emotion(text, model_name)
        return render_template(
            "index.html",
            text=text,
            emotion=emotion,
            model=model_name
        )
    return render_template("index.html")

# --------------------------------------------------
# REST API route
# --------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    model_name = data.get("model", "logreg")
    emotion = predict_emotion(text, model_name)
    return jsonify({
        "input_text": text,
        "model_used": model_name,
        "predicted_emotion": emotion
    })

# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
