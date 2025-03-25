from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel
import base64
from io import BytesIO

app = Flask(__name__)

# API configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCsUMAfUu-IXXpIuJNdEKfUtb-d5kTCqIs")
MEOW_API_KEY = os.getenv("MEOW_API_KEY", "meow-6c568600-ab88-4dd4-a28a-b47bb3d664a0")  # Replace with your actual Meow API key

genai.configure(api_key=GEMINI_API_KEY)
meow_client = OpenAI(base_url="https://meow.cablyai.com/v1", api_key=MEOW_API_KEY)

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
        else:  # Using Meow API
            response = meow_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.7
            )
            return jsonify({"response": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

# New endpoint for processing image with text query
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
        
        return jsonify({"response": response.text})

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