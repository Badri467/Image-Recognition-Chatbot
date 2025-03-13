# Image Recognition Chatbot

An intelligent chatbot application that combines advanced image recognition capabilities with multiple language models to analyze images and respond to user queries.

## Features

- **Image Analysis**: Upload images and ask specific questions about their content
- **Visual Concept Extraction**: Uses the CLIP model to identify objects, activities, emotions, and styles in uploaded images
- **Enhanced Prompting**: Combines visual concepts with user queries to provide more accurate and contextually relevant responses
- **User-Friendly Interface**: Clean, intuitive chat interface with image upload capabilities
- **Persistent Context**: Ability to ask multiple questions about the same uploaded image

## Tech Stack

### Backend
- **Flask**: Lightweight web framework for Python
- **Google Generative AI**: API for accessing Gemini models
- **Zuki Journey API**: Interface for accessing multiple AI models
- **Hugging Face Transformers**: Used for CLIP model implementation
- **PyTorch**: Deep learning framework for CLIP model inference
- **Pillow (PIL)**: Image processing library

### Frontend
- **HTML/CSS**: For responsive user interface
- **JavaScript**: For client-side interactions
- **Fetch API**: For asynchronous communication with the backend
  

## Use Cases

- **Education**: Analyze diagrams, charts, or educational content in images
- **Research**: Extract information from visual data or research materials
- **Accessibility**: Help visually impaired users understand image content
- **Content Analysis**: Identify objects, themes, and context in photographs
- **Technical Support**: Analyze screenshots of errors or technical issues
- **Travel**: Identify landmarks, locations, or points of interest in travel photos
- **Art Analysis**: Understand artistic styles, techniques, or subject matter in artwork
- **Design Feedback**: Get AI insights on design work, UI mockups, or prototypes

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### API Keys
You'll need to obtain API keys for:
- Google Gemini API
- Zuki Journey API

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image-Recognition-Chatbot.git
cd Image-Recognition-Chatbot
```

2. Install dependencies:
```bash
pip install flask google-generativeai openai pillow torch transformers
```

3. Set up environment variables for your API keys:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export ZUKI_API_KEY="your_zuki_api_key_here"
```

Alternatively, you can directly update the keys in the `app.py` file (not recommended for production).

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## How to Use

1. **Text-Only Queries**:
   - Select the desired AI model from the dropdown menu
   - Type your question in the text field
   - Click "Send" or press Enter

2. **Image Analysis**:
   - Click "Upload" to select an image from your device
   - Type a question about the image in the text field
   - Click "Send" or press Enter
   - Continue asking multiple questions about the same image as needed
   - Click the "x" on the image preview to remove it when finished

## Future Improvements

- User authentication and conversation history
- Support for video analysis
- Batch image processing
- Fine-tuning models for specific domains or use cases
- Image editing or generation capabilities
- Mobile application version


## Acknowledgments

- OpenAI for the CLIP model
- Google for the Gemini API
- Zuki Journey for their API access
