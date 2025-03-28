from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from openai import OpenAI
from groq import Groq
from PIL import Image
import os
from dotenv import load_dotenv
import torch # type: ignore
from transformers import CLIPProcessor, CLIPModel
import base64
from io import BytesIO
import markdown
from markdown.extensions import fenced_code, tables, nl2br
from markdown.extensions.codehilite import CodeHiliteExtension

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure markdown with extensions
md = markdown.Markdown(extensions=[
    'fenced_code',
    'tables',
    'nl2br',
    CodeHiliteExtension(
        css_class='highlight',
        linenums=True,
        linenostart=1,
        use_pygments=True
    )
])

# API configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MEOW_API_KEY = os.getenv("MEOW_API_KEY")
ZUKI_API_KEY = os.getenv("ZUKI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize API clients
genai.configure(api_key=GEMINI_API_KEY)
meow_client = OpenAI(base_url="https://meow.cablyai.com/v1", api_key=MEOW_API_KEY)
zuki_client = OpenAI(base_url="https://api.zuki.ai/v1", api_key=ZUKI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Model configurations
VISION_MODEL = "gemini-1.5-flash"
AVAILABLE_MODELS = {
    "gemini-1.5-flash": "gemini",
    "gpt-4o": "meow",
    "gemini-2.0-flash": "meow",
    "gpt-4o-mini": "meow",
    "grok-2": "meow",
    "llama-3.1-405b": "meow",
    "mistral-nemo-12b": "meow",
    "claude-3.5-sonnet": "meow",
    "claude-3.7-sonnet": "meow",
    "gpt-4.5-preview": "meow",
    "llama-3.2-11b-instruct": "meow",
    "deepseek-coder-6.7b-instruct-awq": "meow",
}

AVAILABLE_MODELS_ZUKI = {
    "gemini-1.5-flash": "gemini",
    "caramelldansen-1": "zuki",
    "claude-3-haiku": "zuki",
    "deepseek-coder-6.7b-base": "zuki",
    "deepseek-coder-6.7b-instruct": "zuki"
}

AVAILABLE_MODELS_GROQ = {
    "distil-whisper-large-v3-en": "groq",
    "genuna2-9b-it": "groq",
    "llama-3.3-70b-versatile": "groq",
    "llama-3.1-8b-instant": "groq",
    "llama-guard-3-8b": "groq",
    "llama3-70b-8192": "groq",
    "llama3-8b-8192": "groq",
    "whisper-large-v3": "groq",
    "whisper-large-v3-turbo": "groq"
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
        provider = data.get("provider", "zuki")

        if not user_query:
            return jsonify({"response": "Please enter a valid query."}), 400

        response_text = ""
        if provider == "zuki":
            if selected_model in AVAILABLE_MODELS_ZUKI:
                if AVAILABLE_MODELS_ZUKI[selected_model] == "gemini":
                    # Handle Gemini model
                    model = genai.GenerativeModel(selected_model)
                    response = model.generate_content(user_query)
                    response_text = response.text
                else:
                    # Handle other Zuki models
                    response = zuki_client.chat.completions.create(
                        model=selected_model,
                        messages=[{"role": "user", "content": user_query}]
                    )
                    response_text = response.choices[0].message.content
            else:
                return jsonify({"response": "Selected model is not available for Zuki provider."}), 400
        elif provider == "groq":
            if selected_model in AVAILABLE_MODELS_GROQ:
                response = groq_client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
            else:
                return jsonify({"response": "Selected model is not available for Groq provider."}), 400
        else:  # Using Meow API
            if selected_model in AVAILABLE_MODELS:
                response = meow_client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
            else:
                return jsonify({"response": "Selected model is not available for Meow provider."}), 400

        # Convert markdown to HTML and add model name
        html_content = md.convert(response_text)
        formatted_response = f"{html_content}\n\n---\nModel: {selected_model}"
        return jsonify({"response": formatted_response})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route("/process_image_query", methods=["POST"])
def process_image_query():
    try:
        # Get the query text and image from the request
        query = request.form.get("query", "")
        if "image" not in request.files:
            return jsonify({"response": "No image provided."}), 400
            
        file = request.files["image"]
        image = Image.open(file)
        
        # Use CLIP to extract visual concepts
        visual_concepts = extract_visual_concepts(image)
        
        # Create context from visual concepts for Gemini
        concepts_str = ", ".join([concept for concept, _ in visual_concepts[:10]])
        
        # Build enhanced prompt with CLIP concepts and user query
        enhanced_prompt = f"""
        I've analyzed this image and detected: {concepts_str}.
        
        User's question about this image: "{query}"
        
        Please provide a detailed response that addresses the user's question while incorporating relevant visual elements from the image.
        """
        
        # Process with Gemini Vision model
        model = genai.GenerativeModel(VISION_MODEL)
        response = model.generate_content([enhanced_prompt, image])
        
        # Convert markdown to HTML and add model name
        html_content = md.convert(response.text)
        formatted_response = f"{html_content}\n\n---\nModel: {VISION_MODEL}"
        return jsonify({"response": formatted_response})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

# Function to extract visual concepts using CLIP
def extract_visual_concepts(image):
    # Predefined categories to check in the image
    categories = [
        "person", "animal", "dog", "cat", "bird", "vehicle", "car", "building", 
        "landscape", "mountain", "beach", "forest", "city", "indoor", "outdoor",
        "day", "night", "food", "drink", "technology", "art", "text", "water",
        "sky", "cloudy", "sunny", "group of people", "furniture", "plants", "flowers"
    ]
    
    # Activities
    activities = [
        "walking", "running", "sitting", "standing", "eating", "drinking", "playing",
        "working", "reading", "writing", "cooking", "dancing", "singing", "swimming"
    ]
    
    # Emotions/styles
    styles = [
        "happy", "sad", "colorful", "minimalist", "vintage", "modern", "natural",
        "artificial", "professional", "casual", "artistic", "technical", "bright", "dark"
    ]
    
    # Combine all concept categories
    all_concepts = categories + activities + styles
    
    # Prepare inputs for CLIP
    inputs = clip_processor(
        text=all_concepts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Get model outputs
    with torch.no_grad():
        outputs = clip_model(**inputs)
    
    # Calculate similarities
    logits_per_image = outputs.logits_per_image
    probs = torch.nn.functional.softmax(logits_per_image, dim=1)
    
    # Filter concepts with probability above threshold
    threshold = 0.15
    detected_concepts = []
    for i, concept in enumerate(all_concepts):
        probability = probs[0][i].item()
        if probability > threshold:
            detected_concepts.append((concept, probability))
    
    # Sort by probability
    detected_concepts.sort(key=lambda x: x[1], reverse=True)
    
    return detected_concepts

if __name__ == "__main__":
    app.run(debug=True)
