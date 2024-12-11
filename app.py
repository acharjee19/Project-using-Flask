from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import torch

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load the Hugging Face translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fi", device=device)

@app.route("/", methods=["GET"])
def home():
    # Render the HTML template
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Preprocess input
        text = text.strip()
        # Perform translation
        translated_text = translator(text, max_length=100)[0]["translation_text"]
        print("Input Text:", text)  # Debugging
        print("Translated Text:", translated_text)  # Debugging
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Debugging: Print URL mapping
print(app.url_map)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
