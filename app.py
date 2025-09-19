from flask import Flask, request, jsonify, render_template
import joblib

# Load models and preprocessing tools
log_reg = joblib.load("models/logistic_regression_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Flask app
app = Flask(__name__)

def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def predict_emotion(text, model_name="logreg"):
    cleaned = preprocess_text(text)
    seq = vectorizer.transform([cleaned])

    if model_name == "logreg":
        pred = log_reg.predict(seq)[0]
    elif model_name == "svm":
        pred = svm_model.predict(seq)[0]
    else:
        return None

    emotion = label_encoder.inverse_transform([pred])[0]
    return emotion

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        model_name = request.form.get("model", "logreg")
        emotion = predict_emotion(text, model_name)
        return render_template("index.html", text=text, emotion=emotion, model=model_name)
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = data.get("text", "")
    model_name = data.get("model", "logreg")
    emotion = predict_emotion(text, model_name)
    return jsonify({"text": text, "model": model_name, "predicted_emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
