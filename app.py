from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from PIL import Image
import os

app = Flask(__name__)


API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
genai.configure(api_key=API_KEY)


VISION_MODEL = "gemini-1.5-flash" 
TEXT_MODEL = "gemini-1.5-flash"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def text_query():
    try:
        data = request.json
        user_query = data.get("query", "")

        if not user_query:
            return jsonify({"response": "Please enter a valid query."}), 400

        model = genai.GenerativeModel(TEXT_MODEL)
        response = model.generate_content(user_query)
        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"response": "No image provided."}), 400

        file = request.files["image"]
        image = Image.open(file)

       
        model = genai.GenerativeModel(VISION_MODEL)

      
        response = model.generate_content([
            "Analyze this image and describe its contents in detail",
            image
        ])

        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
