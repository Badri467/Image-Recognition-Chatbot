# 🚀 OmniSense AI - Advanced Multi-Modal Intelligent Platform

OmniSense AI is a sophisticated Flask-based AI platform that combines multiple cutting-edge capabilities including multi-model text generation, image analysis, voice interaction, real-time collaboration, and dynamic animation generation. This platform seamlessly integrates several state-of-the-art AI technologies to provide a comprehensive communication and creation tool.

![chatbot interface](images/Screenshot%202025-05-13%20085236.png)
![image analysis](images/Screenshot%202025-03-29%20020211.png)
![image analysis](images/Screenshot%202025-03-29%20020155.png)
![models available](images/Screenshot%202025-03-29%20020047.png)
![2d animation videos generation](images/Screenshot%202025-05-13%20082343.png)
![web search results](images/Screenshot%202025-04-18%20184331.png)
A full-featured AI chatbot platform built with Flask that integrates multiple LLM services:

- 🤖 Multiple model support (Gemini, Groq, OpenRouter)
- 🔍 Web search augmentation for up-to-date information
- 🖼️ Image recognition and visual content analysis
- 🎤 Voice chat with Eleven Labs text-to-speech
- 👨‍👩‍👧‍👦 Collaborative shared chat sessions
- 📊 Animation generation with Manim
- 🔒 User authentication system
- 📱 Real-time communications with Socket.IO

Perfect for developers looking to build advanced AI chat applications with modern features.
## 🌟 Key Features

### 🧠 Advanced Multi-Model AI Interaction
- Support for multiple AI models:
  - **Gemini 2.0 Flash** - Google's advanced text and vision model
  - **Llama 3.3 70B** - High-capacity language model
  - **Llama 3.1 8B** - Faster, smaller model for quicker responses
  - **Various other models** via Groq API integration

### 🔍 Enhanced Web Search Integration
- Real-time Google Search API integration for current information
- Intelligent query analysis to determine when to use search enhancement
- Web content extraction and summarization
- Properly cited sources with clickable references
- Contextual information integration into AI responses

### 👁️ Computer Vision & Image Analysis
- CLIP (Contrastive Language-Image Pre-training) integration for image understanding
- Automatic visual concept extraction from uploaded images
- Sophisticated image-based querying and analysis
- Multi-modal comprehension combining visual and textual inputs

### 🎬 Real-time 2D Animation Generation
- Custom Manim CE integration for mathematical and conceptual animations
- Text-to-animation capabilities
- Dynamic visualization of complex concepts
- Real-time rendering and display in browser

### 🔊 Voice Interaction System
- Text-to-speech using ElevenLabs advanced voice synthesis
- Multiple voice options with customizable settings
- Natural-sounding voice output for accessibility
- Speech-optimized text processing

### 👥 Real-time Collaborative Workspace
- Socket.IO-powered real-time chat rooms
- Shared chat sessions with unique shareable codes
- Multi-user synchronized interactions
- Live user presence indicators
- Guest access system with temporary accounts

### 🔒 User Authentication System
- Secure user registration and login
- Password hashing for security
- Session management
- Guest account provisioning for quick access

### 📊 Markdown & Code Rendering
- Advanced markdown processing with syntax highlighting
- Code block language detection
- Beautiful formatting of complex responses
- Support for tables, lists, and other formatting elements

## 🛠️ Technical Implementation

### Backend Architecture
- **Flask**: Core web framework with extensive routing
- **MongoDB**: Document database for chat storage and user management
- **Socket.IO**: Real-time bidirectional event-based communication
- **APScheduler**: Background task scheduling
- **Waitress**: Production-grade WSGI server

### AI Service Integration
- **Google Generative AI**: For Gemini model access
- **Groq**: For Llama model access
- **OpenAI/OpenRouter**: For additional model options
- **ElevenLabs**: For text-to-speech capabilities

### Vision Processing
- **CLIP**: For image understanding and concept extraction
- **Transformers**: Hugging Face libraries for model management
- **PIL**: For image processing and manipulation

### Animation System
- **Manim CE**: Mathematical Animation Engine
- **Dynamic code generation**: AI-generated animation scripts
- **Subprocess management**: For rendering animations

### Collaborative Features
- **Room-based chat system**: For multiple collaborative spaces
- **Shared state management**: For consistent user experiences
- **Presence indicators**: For active user awareness

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MongoDB
- Manim CE (for animation features)
- API keys for: Google Generative AI, Groq, Google Search, ElevenLabs

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/omnisense-ai.git
cd omnisense-ai
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys
```
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_preferred_voice_id
MONGO_URI=your_mongodb_connection_string
FLASK_SECRET_KEY=your_secret_key
```

5. Run the application
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## 💡 Use Cases

### Educational Tools
- Generate explanatory animations for complex concepts
- Real-time Q&A with web search augmentation
- Collaborative learning environments

### Content Creation
- Quick animation generation for presentations
- Voice synthesis for narration
- Image analysis and description

### Team Collaboration
- Shared AI-assisted workspaces
- Real-time collaborative problem-solving
- Knowledge sharing with integrated search

### Research Assistance
- Web-augmented information gathering
- Visual data analysis through image uploads
- Complex query processing with multiple AI models

## 🌐 Future Enhancements

- Mobile application integration
- Custom fine-tuned models
- Advanced document analysis
- More animation templates and styles
- Audio input processing
- Integration with popular productivity tools



Created with ❤️ by [K.BadriNaryana]

*Note: This platform requires valid API keys for full functionality. Some features may require additional configuration.*
