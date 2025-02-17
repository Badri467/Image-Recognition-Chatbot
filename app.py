from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import os

app = Flask(__name__)

# API configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
ZUKI_API_KEY = os.getenv("ZUKI_API_KEY", "YOUR_ZUKI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
zuki_client = OpenAI(base_url="https://api.zukijourney.com/v1", api_key=ZUKI_API_KEY)

# Model configurations
VISION_MODEL = "gemini-1.5-flash"  # Keep Gemini for vision tasks
AVAILABLE_MODELS = {
    "gemini-1.5-flash": "gemini",
    "caramelldansen-1": "zuki",
    "gpt-4-mini": "zuki",
    "claude-3-haiku": "zuki",
    "deepseek-coder-6.7b-base": "zuki",
    "deepseek-coder-6.7b-instruct": "zuki"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def text_query():
    try:
        data = request.json
        user_query = data.get("query", "")
        selected_model = data.get("model", "gemini-1.5-flash")

        if not user_query:
            return jsonify({"response": "Please enter a valid query."}), 400

        if AVAILABLE_MODELS[selected_model] == "gemini":
            model = genai.GenerativeModel(selected_model)
            response = model.generate_content(user_query)
            return jsonify({"response": response.text})
        else:
            response = zuki_client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": user_query}]
            )
            return jsonify({"response": response.choices[0].message.content})

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
