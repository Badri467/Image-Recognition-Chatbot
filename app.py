
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
import tempfile
from google.generativeai import types
import mimetypes
import re

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
        use_pygments=True,
        noclasses=False,
        pygments_style='monokai',
        guess_lang=True,
        lang_prefix='hljs language-',
        pygments_lang_class=True,
        use_pygments_style=True
    )
])

# API configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize API clients
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Create cache directory for models
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

# Load CLIP model with caching
try:
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=cache_dir,
        local_files_only=True
    )
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=cache_dir,
        local_files_only=True
    )
except Exception:
    # If model not found locally, download it once
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=cache_dir
    )
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=cache_dir
    )

# Model configurations
VISION_MODEL = "gemini-1.5-flash"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def text_query():
    try:
        data = request.json
        user_query = data.get("query", "")
        selected_model = data.get("model", "gemini-1.5-flash")
        temperature = data.get("temperature", 0.7)

        if not user_query:
            return jsonify({"response": "Please enter a valid query."}), 400

        response_text = ""
        # Handle Gemini model
        if selected_model == "gemini-1.5-flash":
            model = genai.GenerativeModel(selected_model)
            response = model.generate_content(user_query)
            response_text = response.text
            
        # Handle Groq models
        elif selected_model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", 
                               "llama3-70b-8192", "llama3-8b-8192"]:
            response = groq_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_query}
                ],
                temperature=temperature
            )
            response_text = response.choices[0].message.content
            
        # Handle OpenRouter models
        elif selected_model in ["deepseek/deepseek-v3-base:free", 
                               "qwen/qwen2.5-vl-3b-instruct:free"]:
            if not OPENROUTER_API_KEY:
                return jsonify({"response": "OpenRouter API key not configured."}), 500

            openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY
            )
            
            try:
                response = openrouter_client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=temperature,
                    extra_headers={
                        "HTTP-Referer": "http://localhost:5000",
                        "X-Title": "Image Recognition Chatbot"
                    }
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                return jsonify({"response": f"OpenRouter Error: {str(e)}"}), 500

        else:
            return jsonify({"response": "Selected model is not available."}), 400

        # Convert markdown to HTML
        html_content = md.convert(response_text)
        
        # Process code blocks to ensure proper language classes
        def process_code_blocks(html):
            # Find all code blocks
            code_blocks = re.finditer(r'<pre><code(?: class="([^"]*)")?>(.*?)</code></pre>', html, re.DOTALL)
            
            for match in code_blocks:
                original = match.group(0)
                classes = match.group(1) or ''
                content = match.group(2)
                
                # If no language class is specified, try to detect it from the content
                if 'language-' not in classes:
                    # Common language indicators
                    if 'def ' in content or 'import ' in content or 'class ' in content:
                        classes = 'hljs language-python'
                    elif 'function' in content or 'const ' in content or 'let ' in content:
                        classes = 'hljs language-javascript'
                    elif 'public class ' in content or 'import java.' in content or 'System.out.println' in content:
                        classes = 'hljs language-java'
                    elif 'package main' in content or 'import "' in content or 'func ' in content:
                        classes = 'hljs language-go'
                    elif '#include <' in content or 'namespace ' in content or 'std::' in content:
                        classes = 'hljs language-cpp'
                    elif '#include <' in content or 'int main()' in content or 'printf(' in content:
                        classes = 'hljs language-c'
                    elif 'SELECT' in content.upper() or 'INSERT' in content.upper() or 'FROM' in content.upper():
                        classes = 'hljs language-sql'
                    elif '' in content:
                        classes = 'hljs language-markdown'
                    elif '<html' in content or '<body' in content or '<head' in content:
                        classes = 'hljs language-html'
                    elif '{' in content and '}' in content:
                        classes = 'hljs language-css'
                    else:
                        classes = 'hljs language-plaintext'
                
                # Replace the original code block with the processed version
                new_block = f'<pre><code class="{classes}">{content}</code></pre>'
                html = html.replace(original, new_block)
            
            return html
        
        # Process the HTML content
        html_content = process_code_blocks(html_content)
        
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
        enhanced_prompt = f """
        I've analyzed this image and detected: {concepts_str}.
        
        User's question about this image: "{query}"
        
        Please provide a detailed response that addresses the user's question while incorporating relevant visual elements from the image.
        """
        
        # Process with Gemini Vision model
        model = genai.GenerativeModel(VISION_MODEL)
        response = model.generate_content([enhanced_prompt, image])
        
        # Convert markdown to HTML and format code blocks
        html_content = md.convert(response.text)
        html_content = html_content.replace('<pre><code>', '<div class="code-block"><pre><code>')
        html_content = html_content.replace('</code></pre>', '</code></pre></div>')
        
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
