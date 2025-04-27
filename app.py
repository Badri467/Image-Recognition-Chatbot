from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response
import google.generativeai as genai
from flask_jwt_extended import jwt_required
from openai import OpenAI
from groq import Groq
from PIL import Image
import requests
from bs4 import BeautifulSoup
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
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from flask_socketio import SocketIO, join_room, leave_room, emit
import uuid
import json
import secrets
from flask_cors import CORS
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from waitress import serve

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")  # for session management
CORS(app, supports_credentials=True)
# Initialize Eleven Labs client
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Default voice and model settings for Eleven Labs
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  
DEFAULT_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")  

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client.chatbot_db

socketio = SocketIO(app, 
                   cors_allowed_origins="*",  
                   ping_timeout=60,
                   ping_interval=25,
                   async_mode='threading')  # Use threading mode for better handling of multiple clients

# Create a collection for shared chats
shared_chats = db.shared_chats
# Collections
chats = db.chats
users = db.users
current_date = datetime.now().strftime('%A, %B %d, %Y')
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12)
)
# Helper function to create new chat
def create_new_chat():
    try:
        if 'user_id' not in session:
            return None  # Or handle unauthorized access
        
        chat_data = {
            'created_at': datetime.utcnow(),
            'messages': [],
            'user_id': ObjectId(session['user_id']),
            'title': 'New Chat'
        }
        
        # Insert the new chat and return its ID
        result = chats.insert_one(chat_data)
        return str(result.inserted_id)
    
    except Exception as e:
        print(f"Error creating chat: {str(e)}")
        return None

# Helper function to save message
def save_message(chat_id, role, content, model=None, image=None):
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.utcnow(),
        'model': model
    }
    
    if image:
        # Reset file pointer and encode image
        image.seek(0)
        message['image'] = {
            'data': base64.b64encode(image.read()).decode('utf-8'),
            'content_type': image.content_type,
            'filename': image.filename
        }
    
    # Update chat document with the new message
    result = chats.update_one(
        {'_id': ObjectId(chat_id)},
        {
            '$push': {'messages': message},
            '$setOnInsert': {  # Only set these fields when creating new document
                'user_id': ObjectId(session.get('user_id')) if 'user_id' in session else None,
                'created_at': datetime.utcnow()
            }
        },
        upsert=True
    )
    
    return message

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
# Google Search API configuration
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
# Initialize API clients
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Create cache directory for models
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

# Add session configuration
app.config.update(
    SESSION_COOKIE_NAME='chatbot_session',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False,  # Set to True in production
    PERMANENT_SESSION_LIFETIME=86400  # 1 day
)

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
VISION_MODEL = "gemini-2.0-flash"

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register_page():
    return render_template('register.html')

@app.route("/")
def home():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    # Create new chat session if none exists
    if 'chat_id' not in session:
        session['chat_id'] = create_new_chat()
    return render_template("index.html")

@app.route('/login', methods=['POST'])
def login():
    data = request.form
    user = users.find_one({'username': data['username']})
    if user and check_password_hash(user['password'], data['password']):
        session['user_id'] = str(user['_id'])
        session['chat_id'] = create_new_chat()
        return jsonify({'status': 'success'})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.form
    if users.find_one({'username': data['username']}):
        return jsonify({'error': 'Username exists'}), 400
        
    user_id = users.insert_one({
        'username': data['username'],
        'password': generate_password_hash(data['password']),
        'created_at': datetime.utcnow()
    }).inserted_id
    
    session['user_id'] = str(user_id)
    session['chat_id'] = create_new_chat()
    return jsonify({'status': 'success'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# Update new_chat route
@app.route("/new_chat", methods=["POST"])
def new_chat():
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401

        # Create new chat
        new_chat_id = create_new_chat()
        if not new_chat_id:
            return jsonify({"error": "Chat creation failed"}), 500

        # Update session
        session['chat_id'] = new_chat_id
        
        # Update previous chat title
        old_chat_id = session.get('previous_chat_id')
        if old_chat_id:
            try:
                first_message = chats.find_one(
                    {'_id': ObjectId(old_chat_id)},
                    {'messages': {'$slice': 1}}
                )
                if first_message and first_message.get('messages'):
                    title = first_message['messages'][0]['content'][:50] + "..."
                    chats.update_one(
                        {'_id': ObjectId(old_chat_id)},
                        {'$set': {'title': title}}
                    )
            except Exception as update_error:
                print(f"Title update error: {update_error}")

        session['previous_chat_id'] = new_chat_id
        
        return jsonify({
            "status": "success",
            "chat_id": new_chat_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_user_chats")
def get_user_chats():
    if 'user_id' not in session:
        return jsonify([])
        
    user_chats = chats.find({'user_id': ObjectId(session['user_id'])}).sort('created_at', -1)
    return jsonify([{
        '_id': str(chat['_id']),
        'title': chat.get('title', 'New Chat'),
        'created_at': chat['created_at'].isoformat()
    } for chat in user_chats])

@app.route("/get_current_chat")
def get_current_chat():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    if 'chat_id' not in session:
        session['chat_id'] = create_new_chat()
    
    return jsonify({"chat_id": session['chat_id']})

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    chat_id = session.get('chat_id')
    if not chat_id:
        return jsonify([])
    
    chat = chats.find_one({'_id': ObjectId(chat_id)})
    if not chat:
        return jsonify([])
    
    return jsonify(chat['messages'])

@app.route("/query", methods=["POST"])
def text_query():
    try:
        data = request.json
        user_query = data.get("query", "")
        selected_model = data.get("model", "gemini-2.0-flash")
        temperature = data.get("temperature", 0.7)
        chat_id = session.get('chat_id')

        if not user_query:
            return jsonify({"response": "Please enter a valid query."}), 400

        # Save user message
        save_message(chat_id, "user", user_query)
        
        # Determine if search should be used
        use_search = should_use_search(user_query) and data.get("enable_search", True)
        search_results = None
        structured_sources = []
        
        if use_search and GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
            search_results, structured_sources = perform_search(user_query)
            print(f"Search results found: {bool(search_results)}")
        
        response_text = ""
        
        # Handle Gemini model with search augmentation
        if selected_model == "gemini-2.0-flash":
            model = genai.GenerativeModel(selected_model)
            current_date = datetime.now().strftime('%A, %B %d, %Y')
            
            if search_results:
                prompt = f"""You are an intelligent assistant. Today's date is {current_date}.
                
                Question: {user_query}
                
                Based on the following web search results, provide a comprehensive answer:
                
                {search_results}
                
                Synthesize the information from these sources to answer the question accurately.
                If the search results contain relevant information, incorporate it into your response.
                If the search results don't provide enough information, use your knowledge to supplement.
                Always cite the source numbers when referencing information from the search results (e.g., (1), (2)).
                Do not include the actual URLs in your main response text.
                """
                response = model.generate_content(prompt)
            else:
                response = model.generate_content(f"Today's date is {current_date}.\n\nQuestion: {user_query}")
                
            response_text = response.text
            
        # Handle Groq models
        elif selected_model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", 
                       "llama3-70b-8192", "llama3-8b-8192"]:
    
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            
            # Add search results if available
            if search_results:
                # Extract URLs for references
                search_urls = []
                for line in search_results.split('\n'):
                    if line.startswith('URL:'):
                        search_urls.append(line.replace('URL:', '').strip())
                
                sources_list = "\n".join([f"[{i+1}] {url}" for i, url in enumerate(search_urls)])
                
                system_msg = f"""Use these search results when helpful: {search_results}
                
                IMPORTANT: Cite sources with numbers (e.g., (1), (2)) when using information from search results.
                IMPORTANT: Include all reference URLs in a 'References' section at the end of your response, like this:
                
                References:
                {sources_list}
                """
                messages.append({"role": "system", "content": system_msg})
            
            messages.append({"role": "user", "content": user_query})
                
            response = groq_client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature
            )
            response_text = response.choices[0].message.content
            
        # Handle OpenRouter models with similar search augmentation pattern
        elif selected_model in ["deepseek/deepseek-v3-base:free", 
                               "qwen/qwen2.5-vl-3b-instruct:free"]:
            if not OPENROUTER_API_KEY:
                return jsonify({"response": "OpenRouter API key not configured."}), 500

            openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY
            )
            
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            
            # Add search results if available
            if search_results:
                messages.append({"role": "system", "content": f"Use these search results when helpful: {search_results}"})
                
            messages.append({"role": "user", "content": user_query})
            
            try:
                response = openrouter_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
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
        
        
        # Process code blocks (keep your existing code)
        html_content = process_code_blocks(html_content)
        
        # Add search indicators and source links if search was used
        if search_results and structured_sources:
            search_indicator = '<div class="search-indicator">🔍 Enhanced with search results</div>'
            
            # Create reference section with clickable links
            references_html = '<div class="references-section"><h4>References</h4><ul>'
            for i, source in enumerate(structured_sources):
                references_html += f'<li><a href="{source["url"]}" target="_blank" class="source-link">{source["title"]}</a></li>'
            references_html += '</ul></div>'
            
            html_content = search_indicator + html_content + references_html
        
        # Save bot response
        save_message(chat_id, "bot", response_text, selected_model)
        
        formatted_response = f"{html_content}\n\n---\nModel: {selected_model}"
        return jsonify({"response": formatted_response})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

    
# New route for text-to-speech using Eleven Labs
@app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    try:
        data = request.json
        text = data.get("text", "")
        voice_id = data.get("voice_id", DEFAULT_VOICE_ID)
        model_id = data.get("model_id", DEFAULT_MODEL_ID)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        # Clean up text for better speech quality
        # Remove markdown code blocks, URLs, and other non-verbal elements
        cleaned_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
        cleaned_text = re.sub(r'\[.*?\]\(.*?\)', '', cleaned_text)  # Remove markdown links
        cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)  # Remove URLs
        cleaned_text = re.sub(r'[\*\_\~\`\#\>\|\-\=\+]', '', cleaned_text)  # Remove special characters
        cleaned_text = cleaned_text.replace('\n\n', '. ').replace('\n', '. ')  # Replace newlines with periods for better flow
        
        # Generate speech with Eleven Labs
        audio = elevenlabs_client.text_to_speech.convert(
            text=cleaned_text,
            voice_id=voice_id,
            model_id=model_id,
            output_format="mp3_44100_128",
        )
        
        # Return audio as response
        return Response(audio, mimetype="audio/mpeg")
        
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        return jsonify({"error": f"Text-to-speech error: {str(e)}"}), 500



@app.route("/process_image_query", methods=["POST"])
def process_image_query():
    try:
        chat_id = session.get('chat_id')
        if not chat_id:
            return jsonify({"error": "No active chat session"}), 400

        # Save user's image message first
        file = request.files["image"]
        file.seek(0)  # Reset file pointer
        save_message(chat_id, "user", request.form.get("query", ""), image=file)

        # Rest of your existing processing code
        image = Image.open(file)
        visual_concepts = extract_visual_concepts(image)
        concepts_str = ", ".join([concept for concept, _ in visual_concepts[:10]])
        
        enhanced_prompt = f"""
        I've analyzed this image and detected: {concepts_str}.
        User's question about this image: "{request.form.get('query', '')}"
        Please provide a detailed response that addresses the user's question while incorporating relevant visual elements from the image.
        """
        
        model = genai.GenerativeModel(VISION_MODEL)
        response = model.generate_content([enhanced_prompt, image])
        
        html_content = md.convert(response.text)
        html_content = html_content.replace('<pre><code>', '<div class="code-block"><pre><code>')
        html_content = html_content.replace('</code></pre>', '</code></pre></div>')
        
        formatted_response = f"{html_content}\n\n---\nModel: {VISION_MODEL}"
        
        # Save bot response
        save_message(chat_id, "bot", formatted_response, VISION_MODEL)
        
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

@app.route("/voice_chat")
def voice_chat():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    return render_template("voice_chat.html")

# Return available Eleven Labs voices
@app.route("/get_voices", methods=["GET"])
def get_voices():
    try:
        voices = elevenlabs_client.voices.get_all()
        return jsonify({
            "voices": [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "description": voice.description
                }
                for voice in voices.voices
            ]
        })
    except Exception as e:
        return jsonify({"error": f"Error getting voices: {str(e)}"}), 500
@app.route("/create_shared_chat", methods=["POST"])
def create_shared_chat():
    if 'user_id' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    try:
        # Generate a unique share code
        share_code = str(uuid.uuid4())[:8]
        
        # Create the shared chat
        chat_data = {
            'created_at': datetime.utcnow(),
            'messages': [],
            'creator_id': ObjectId(session['user_id']),
            'participants': [ObjectId(session['user_id'])],
            'share_code': share_code,
            'title': 'Shared Chat'
        }
        
        result = shared_chats.insert_one(chat_data)
        shared_chat_id = str(result.inserted_id)
        
        # Add user to room
        session['current_shared_chat'] = shared_chat_id
        
        return jsonify({
            "status": "success",
            "chat_id": shared_chat_id,
            "share_code": share_code
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to join a shared chat
@app.route("/join_shared_chat/<share_code>", methods=["GET"])
def join_shared_chat(share_code):
    if 'user_id' not in session:
        return redirect(url_for('handle_guest', share_code=share_code))
    
    try:
        # Find the chat with the given share code
        chat = shared_chats.find_one({"share_code": share_code})
        
        if not chat:
            return jsonify({"error": "Invalid share code"}), 404
        
        # Add user to participants if not already there
        if ObjectId(session['user_id']) not in chat['participants']:
            shared_chats.update_one(
                {"_id": chat['_id']},
                {"$push": {"participants": ObjectId(session['user_id'])}}
            )
        
        # Set the current shared chat in session
        session['current_shared_chat'] = str(chat['_id'])
        
        return jsonify({
            "status": "success",
            "chat_id": str(chat['_id']),
            "title": chat.get('title', 'Shared Chat')
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get shared chat data
@app.route("/get_shared_chat/<chat_id>", methods=["GET"])
def get_shared_chat(chat_id):
    if 'user_id' not in session:
        return jsonify({"error": "Authentication required"}), 401
    
    try:
        chat = shared_chats.find_one({"_id": ObjectId(chat_id)})
        
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        # Check if user is a participant
        if ObjectId(session['user_id']) not in chat['participants']:
            return jsonify({"error": "Not authorized to access this chat"}), 403
        
        # Get usernames of participants
        participant_ids = [str(p) for p in chat['participants']]
        participants = []
        for user_id in participant_ids:
            user = users.find_one({"_id": ObjectId(user_id)})
            if user:
                participants.append({
                    "id": user_id,
                    "username": user.get("username", "Unknown")
                })
        
        return jsonify({
            "chat_id": str(chat['_id']),
            "title": chat.get('title', 'Shared Chat'),
            "share_code": chat['share_code'],
            "participants": participants,
            "messages": chat['messages']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Socket.IO event handlers
@socketio.on('join')
def on_join(data):
    try:
        room = data['room']
        username = data.get('username', 'Anonymous')
        
        # Join the room
        join_room(room)
        
        # Notify others that user has joined
        emit('user_joined', {
            'user_id': session.get('user_id'),
            'username': username,
            'timestamp': datetime.utcnow().isoformat()
        }, room=room)
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('leave')
def on_leave(data):
    room = data['room']
    username = data.get('username', 'Anonymous')
    
    leave_room(room)
    emit('user_left', {
        'user_id': session.get('user_id'),
        'username': username,
        'timestamp': datetime.utcnow().isoformat()
    }, room=room)

@socketio.on('new_message')
def on_new_message(data):
    room = data['room']
    message = data['message']
    username = data.get('username', 'Anonymous')
    user_id = session.get('user_id')
    
    # Save message to database
    try:
        shared_chats.update_one(
            {'_id': ObjectId(room)},
            {'$push': {'messages': {
                'user_id': ObjectId(user_id) if user_id else None,
                'username': username,
                'content': message,
                'timestamp': datetime.utcnow()
            }}}
        )
        
        # Broadcast message to room
        emit('message', {
            'user_id': user_id,
            'username': username,
            'content': message,
            'timestamp': datetime.utcnow().isoformat()
        }, room=room)
    except Exception as e:
        emit('error', {'message': str(e)})

# Modify the query route to support shared chats
@app.route("/shared_query", methods=["POST"])
def shared_query():
    try:
        data = request.json
        user_query = data.get("query", "")
        selected_model = data.get("model", "gemini-2.0-flash")
        temperature = data.get("temperature", 0.7)
        chat_id = data.get("chat_id")
        username = data.get("username", "Anonymous")
        
        if not user_query:
            return jsonify({"response": "Please enter a valid query."}), 400
            
        if not chat_id:
            return jsonify({"response": "Chat ID required"}), 400
        
        # Determine if search should be used
        use_search = should_use_search(user_query)
        search_results = None
        
        if use_search and GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
            search_results = perform_search(user_query)
            
        response_text = ""
        # Handle Gemini model with search augmentation
        if selected_model == "gemini-2.0-flash":
            model = genai.GenerativeModel(selected_model)
            
            # If search results available, augment prompt
            if search_results:
                augmented_prompt = f"""
                Question: {user_query}
                
                Use the following search results to help answer the question, but also use your own knowledge when appropriate:
                
                {search_results}
                
                Based on the search results and your knowledge, please answer the question.
                """
                response = model.generate_content(augmented_prompt)
            else:
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

        html_content = md.convert(response_text)
        # Handle other models as you did before
        if search_results:
            search_indicator = '<div class="search-indicator">🔍 Enhanced with search results</div>'
            html_content = search_indicator + html_content
            
        # Save bot response to shared chat
        shared_chats.update_one(
            {'_id': ObjectId(chat_id)},
            {'$push': {'messages': {
                'role': 'bot',
                'content': response_text,
                'model': selected_model,
                'timestamp': datetime.utcnow(),
                'used_search': bool(search_results)
            }}}
        )
        
        # Emit response via Socket.IO
        socketio.emit('bot_response', {
            'content': html_content,
            'model': selected_model,
            'timestamp': datetime.utcnow().isoformat(),
            'used_search': bool(search_results)
        }, room=chat_id)
        
        formatted_response = f"{html_content}\n\n---\nModel: {selected_model}"
        return jsonify({"response": formatted_response})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500


@app.route('/shared_chat')
def shared_chat():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    return render_template('shared_chat.html')

# Add a route to get user info (needed for shared chat)
@app.route('/get_user_info')
def get_user_info():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    user = users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "user_id": str(user['_id']),
        "username": user['username']
    })


@app.route('/guest_login', methods=['POST'])
def guest_login():
    guest_id = f"guest_{secrets.token_hex(8)}"
    username = f"Guest-{secrets.token_hex(4)}"
    
    user_data = {
        '_id': ObjectId(),
        'username': username,
        'is_guest': True,
        'created_at': datetime.utcnow()
    }
    
    users.insert_one(user_data)
    
    session['user_id'] = str(user_data['_id'])
    session['is_guest'] = True
    
    return jsonify({
        'status': 'success',
        'username': username
    })

@app.route("/handle_guest/<share_code>")
def handle_guest(share_code):
    # Create guest user
    guest_id = f"guest_{secrets.token_hex(8)}"
    username = f"Guest-{secrets.token_hex(4)}"
    
    user_data = {
        '_id': ObjectId(),
        'username': username,
        'is_guest': True,
        'created_at': datetime.utcnow()
    }
    
    users.insert_one(user_data)
    
    session['user_id'] = str(user_data['_id'])
    session['is_guest'] = True
    
    return redirect(url_for('join_shared_chat', share_code=share_code))

def cleanup_guests():
    """Automated cleanup task for guest data"""
    try:
        # Delete guest accounts older than 14 days
        users.delete_many({
            'is_guest': True,
            'created_at': {'$lt': datetime.utcnow() - timedelta(days=14)}
        })
        
        # Delete guest chats older than 7 days
        shared_chats.delete_many({
            'participants': {'$elemMatch': {'is_guest': True}},
            'last_activity': {'$lt': datetime.utcnow() - timedelta(days=7)}
        })
        print(f"Cleanup completed at {datetime.utcnow()}")
    except Exception as e:
        print(f"Cleanup error: {str(e)}")

# Initialize scheduler only once
if not hasattr(app, 'scheduler'):
    scheduler = BackgroundScheduler()
    scheduler.add_job(cleanup_guests, 'interval', hours=24)
    scheduler.start()
    app.scheduler = scheduler  # Attach to app context

def perform_search(query, num_results=3):
    """Enhanced search function that fetches and processes web content"""
    try:
        results = google_search(query, num_results)
        web_contents = []
        structured_results = []  # Store structured results for link generation
        
        for idx, (title, link) in enumerate(results):
            print(f"  - Fetching: {title} ({link})")
            text = extract_text_from_url(link)
            web_contents.append(f"Source {idx+1}: {title}\nURL: {link}\nContent: {text}")
            structured_results.append({"title": title, "url": link})
            
        # Return both formatted content and structured data
        return "\n\n".join(web_contents), structured_results
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return None, []    
def should_use_search(query):
    """
    Determine if the query would benefit from search results
    """
    # Keywords indicating a need for current information
    current_info_keywords = [
        'today', 'latest', 'recent', 'news', 'current', 'updates',
        'weather', 'price', 'stock', 'event', 'happening', 'yesterday',
        'this week', 'this month', 'this year', 'announcement'
    ]
    
    # Patterns indicating specific factual questions
    factual_patterns = [
        r'who is', r'what is', r'when did', r'where is', r'how to', 
        r'why did', r'which', r'whose', r'where can I find', 
        r'how many', r'what are', r'who was'
    ]
    # Update in should_use_search function
    time_sensitive_keywords = ['today', 'current date', 'current time', 'what day is it', 
                            'what is the date', 'election results']
                            
    for keyword in time_sensitive_keywords:
        if keyword in query.lower():
            return True
    # Check for current information keywords
    for keyword in current_info_keywords:
        if keyword in query.lower():
            return True
    
    # Check for factual question patterns
    for pattern in factual_patterns:
        if re.search(pattern, query.lower()):
            return True
    
    # Additional check for specific entities or names (proper nouns)
    words = query.split()
    for i, word in enumerate(words):
        if i > 0 and word[0].isupper():  # Check for capitalized words that aren't first in the sentence
            return True
    
    return False
# Add these functions to app.py
def google_search(query, num_results=3):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_SEARCH_API_KEY,
        'cx': GOOGLE_SEARCH_ENGINE_ID,
        'q': query,
        'num': num_results
    }
    response = requests.get(url, params=params)
    results = response.json().get("items", [])
    return [(item["title"], item["link"]) for item in results]

def extract_text_from_url(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator="\n").strip()[:4000]  # keep content short for Gemini
    except Exception as e:
        return f"Error fetching {url}: {e}"
    
if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
