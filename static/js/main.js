let currentAttachedImage = null;

// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
const modelSelect = document.getElementById('modelSelect');
// Add to main.js
document.getElementById('toggleSidebar').addEventListener('click', () => {
  const sidebar = document.querySelector('.sidebar');
  sidebar.classList.toggle('collapsed');
});
function setTheme(isDark) {
  document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
  themeToggle.textContent = isDark ? '☀️' : '🌓';
}

// Initialize theme
setTheme(prefersDarkScheme.matches);

themeToggle.addEventListener('click', () => {
  const isDark = document.body.getAttribute('data-theme') === 'dark';
  setTheme(!isDark);
});

// Image upload handling
document.getElementById("imageInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  currentAttachedImage = file;

  const reader = new FileReader();
  reader.onload = (event) => {
    document.getElementById("imagePreview").src = event.target.result;
    document.getElementById("attachmentArea").classList.remove("hidden");
  };
  reader.readAsDataURL(file);
});

// Remove attachment handling
document.getElementById("removeAttachment").addEventListener("click", () => {
  currentAttachedImage = null;
  document.getElementById("attachmentArea").classList.add("hidden");
  document.getElementById("imagePreview").src = "";
});

function detectLanguage(code) {
  // Common language indicators with more specific checks
  if (code.includes('public class ') || code.includes('import java.') || code.includes('System.out.println')) {
    return 'java';
  } else if (code.includes('package main') || code.includes('import "') || code.includes('func ')) {
    return 'go';
  } else if (code.includes('#include <') && (code.includes('namespace ') || code.includes('std::'))) {
    return 'cpp';
  } else if (code.includes('#include <') && !code.includes('namespace') && !code.includes('std::')) {
    return 'c';
  } else if (code.includes('def ') || code.includes('import ') || code.includes('class ')) {
    return 'python';
  } else if (code.includes('function') || code.includes('const ') || code.includes('let ')) {
    return 'javascript';
  } else if (code.includes('SELECT') || code.includes('INSERT') || code.includes('FROM')) {
    return 'sql';
  } else if (code.includes('```')) {
    return 'markdown';
  } else if (code.includes('<html') || code.includes('<body') || code.includes('<head')) {
    return 'html';
  } else if (code.includes('{') && code.includes('}')) {
    return 'css';
  }
  return 'plaintext';
}
// Update loadChatList function
async function loadChatList() {
  try {
      const response = await fetch('/get_user_chats');
      if (!response.ok) throw new Error('Failed to load chats');
      
      const chats = await response.json();
      const chatList = document.getElementById('chatList');
      
      chatList.innerHTML = chats.map(chat => `
          <div class="chat-item" data-chat-id="${chat._id}">
              <span>${chat.title}</span>
              <small>${new Date(chat.created_at).toLocaleDateString()}</small>
          </div>
      `).join('');
      
      // Add active state to current chat
      const currentChatId = sessionStorage.getItem('currentChat');
      document.querySelectorAll('.chat-item').forEach(item => {
          if (item.dataset.chatId === currentChatId) {
              item.classList.add('active-chat');
          }
      });
  } catch (error) {
      console.error('Error loading chat list:', error);
  }
}
// Add to main.js
// Login form handling
document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  
  try {
      const response = await fetch('/login', {
          method: 'POST',
          body: formData
      });
      
      const data = await response.json();
      if (response.ok) {
          window.location.href = '/';
      } else {
          alert(data.error);
      }
  } catch (error) {
      console.error('Login error:', error);
  }
});

// Register form handling
document.getElementById('registerForm')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);

  try {
      const response = await fetch('/register', {
          method: 'POST',
          body: formData
      });
      
      const data = await response.json();
      if (response.ok) {
          window.location.href = '/';
      } else {
          alert(data.error);
      }
  } catch (error) {
      console.error('Registration error:', error);
  }
});

// Add to DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  // Check localStorage for currentChat, else fetch from API
  const currentChat = localStorage.getItem('currentChat');
  if (!currentChat) {
    fetch('/get_current_chat')
      .then(response => response.json())
      .then(data => {
        if (data.chat_id) {
          localStorage.setItem('currentChat', data.chat_id);
        }
      })
      .catch(error => console.error('Error fetching current chat:', error));
  }
  
  loadChatList();
  loadChatHistory();
});
function processCodeBlock(block) {
  const pre = block.parentElement;
  const codeBlock = document.createElement('div');
  codeBlock.className = 'code-block';
  
  // Create header
  const header = document.createElement('div');
  header.className = 'header';
  
  // Get or detect language
  let language = block.className.match(/language-(\w+)/)?.[1];
  if (!language) {
    language = detectLanguage(block.textContent);
  }

  // Map language codes to display names
  const languageNames = {
    'python': 'Python',
    'javascript': 'JavaScript',
    'html': 'HTML',
    'css': 'CSS',
    'json': 'JSON',
    'bash': 'Bash',
    'markdown': 'Markdown',
    'yaml': 'YAML',
    'sql': 'SQL',
    'cpp': 'C++',
    'java': 'Java',
    'go': 'Go',
    'c': 'C',
    'plaintext': 'Plain Text'
  };
  
  // Add language label with proper name
  const languageLabel = document.createElement('span');
  languageLabel.textContent = languageNames[language] || language;
  
  // Add copy button
  const copyButton = document.createElement('button');
  copyButton.className = 'copy-button';
  copyButton.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"/>
    </svg>
    Copy code
  `;
  
  header.appendChild(languageLabel);
  header.appendChild(copyButton);
  
  // Create content container with line numbers
  const contentWrapper = document.createElement('div');
  contentWrapper.className = 'content';
  
  // Add line numbers
  const lineNumbers = document.createElement('div');
  lineNumbers.className = 'line-numbers';
  const lines = block.textContent.split('\n');
  lines.forEach((_, i) => {
    const lineNumber = document.createElement('span');
    lineNumber.textContent = (i + 1).toString();
    lineNumbers.appendChild(lineNumber);
  });
  
  // Add code content
  const codeContent = document.createElement('div');
  codeContent.className = 'code-content';
  
  // Apply syntax highlighting
  const code = block.textContent;
  const highlightedCode = hljs.highlight(code, { 
    language: language,
    ignoreIllegals: true
  }).value;
  
  // Create a new pre element with the highlighted code
  const preElement = document.createElement('pre');
  preElement.className = `hljs language-${language}`;
  preElement.innerHTML = highlightedCode;
  
  codeContent.appendChild(preElement);
  
  contentWrapper.appendChild(lineNumbers);
  contentWrapper.appendChild(codeContent);
  
  codeBlock.appendChild(header);
  codeBlock.appendChild(contentWrapper);
  
  // Add copy functionality
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(block.textContent);
      copyButton.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path fill-rule="evenodd" clip-rule="evenodd" d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Copied!
      `;
      setTimeout(() => {
        copyButton.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M7 5C7 3.34315 8.34315 2 10 2H19C20.6569 2 22 3.34315 22 5V14C22 15.6569 20.6569 17 19 17H17V19C17 20.6569 15.6569 22 14 22H5C3.34315 22 2 20.6569 2 19V10C2 8.34315 3.34315 7 5 7H7V5ZM9 7H14C15.6569 7 17 8.34315 17 10V15H19C19.5523 15 20 14.5523 20 14V5C20 4.44772 19.5523 4 19 4H10C9.44772 4 9 4.44772 9 5V7ZM5 9C4.44772 9 4 9.44772 4 10V19C4 19.5523 4.44772 20 5 20H14C14.5523 20 15 19.5523 15 19V10C15 9.44772 14.5523 9 14 9H5Z" fill="currentColor"/>
          </svg>
          Copy code
        `;
      }, 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  });
  
  // Replace the original pre element
  pre.parentNode.replaceChild(codeBlock, pre);
}

function appendMessage(sender, content, isImage = false) {
  const chatBox = document.getElementById("chatBox");
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${sender}`;

  const bubbleDiv = document.createElement("div");
  bubbleDiv.className = "bubble";

  if (isImage) {
    const img = document.createElement("img");
    img.src = content;
    bubbleDiv.appendChild(img);
  } else {
    // Split content into HTML and model name
    const parts = content.split('\n\n---\nModel: ');
    const mainContent = parts[0];
    const modelName = parts[1] || '';

    // Set the HTML content
    bubbleDiv.innerHTML = mainContent;

    // Add model name if present
    if (modelName) {
      const modelSpan = document.createElement('span');
      modelSpan.className = 'model-name';
      modelSpan.textContent = `Model: ${modelName}`;
      bubbleDiv.appendChild(modelSpan);
    }

    // Process code blocks
    const codeBlocks = bubbleDiv.querySelectorAll('pre code');
    codeBlocks.forEach(processCodeBlock);
  }

  messageDiv.appendChild(bubbleDiv);
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function handleSend() {
  const queryInput = document.getElementById('textQuery');
  const query = queryInput.value.trim();
  const model = document.getElementById('modelSelect').value;
  const enableSearch = document.getElementById('enableSearch').checked;

  if (!query) return;

  // Add user message to chat
  appendMessage("user", query);
  
  // If there's an image, add it to the chat
  if (currentAttachedImage) {
    const imageUrl = URL.createObjectURL(currentAttachedImage);
    appendMessage("user", imageUrl, true);
  }
  
  // Clear input field
  queryInput.value = "";

  try {
    if (currentAttachedImage) {
      const formData = new FormData();
      formData.append("image", currentAttachedImage);
      formData.append("query", query);
      
      const response = await fetch('/process_image_query', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      appendMessage("bot", data.response);
    } else {
      const response = await fetch('/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          model: model,
          temperature: parseFloat(temperatureSlider.value),
          enable_search: enableSearch
        })
      });

      const data = await response.json();
      appendMessage("bot", data.response);
    }
  } catch (error) {
    appendMessage("bot", "Error: " + error.message);
  }
}


// Send button click handler
document.getElementById('sendButton').addEventListener('click', handleSend);

// Enter key handler
document.getElementById('textQuery').addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

document.getElementById("uploadButton").addEventListener("click", () => {
  document.getElementById("imageInput").click();
});

// Temperature slider functionality
const temperatureControl = document.getElementById('temperatureControl');
const temperatureSlider = document.getElementById('temperatureSlider');
const temperatureValue = document.getElementById('temperatureValue');

temperatureSlider.addEventListener('input', (e) => {
  temperatureValue.textContent = e.target.value;
});

// Load chat history when page loads
async function loadChatHistory() {
    try {
        const response = await fetch('/get_chat_history');
        const messages = await response.json();
        
        // Clear existing messages
        const chatBox = document.getElementById('chatBox');
        chatBox.innerHTML = '';
        
        // Append each message from history
       // Update message loading to handle images
        messages.forEach(message => {
          if (message.image) {
              // Convert base64 image data to URL
              const imgSrc = `data:${message.image.content_type};base64,${message.image.data}`;
              appendMessage(message.role, imgSrc, true);
          } else {
              appendMessage(message.role, message.content);
          }
          });
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}


// Modified new chat button handler
document.getElementById('newChatButton').addEventListener('click', async () => {
  try {
      const response = await fetch('/new_chat', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
      });
      
      const data = await response.json();
      
      if (!response.ok) {
          throw new Error(data.error || 'Failed to create new chat');
      }

      // Clear UI elements
      const chatBox = document.getElementById('chatBox');
      chatBox.innerHTML = '';
      currentAttachedImage = null;
      document.getElementById("attachmentArea").classList.add("hidden");
      document.getElementById("imagePreview").src = "";
      
      // Add welcome message
      appendMessage("bot", "Welcome to your new chat! How can I help you today?");
      
      // Refresh chat list
      await loadChatList();
      
  } catch (error) {
      console.error('New chat error:', error);
      showErrorToast(error.message);
  }
});

// Add error display function
function showErrorToast(message) {
  const toast = document.createElement('div');
  toast.className = 'error-toast';
  toast.textContent = message;
  document.body.appendChild(toast);
  
  setTimeout(() => {
      toast.remove();
  }, 5000);
}
// Load chat history when page loads
document.addEventListener('DOMContentLoaded', loadChatHistory);

document.addEventListener('DOMContentLoaded', () => {
  // Configure Highlight.js
  hljs.configure({
    languages: ['javascript', 'python', 'html', 'css', 'json', 'bash', 'markdown', 'yaml', 'sql', 'cpp', 'java', 'go', 'c'],
    ignoreUnescapedHTML: true,
    detectLanguage: true
  });

  // Initialize Highlight.js on new messages
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node.nodeType === 1) { // Element node
          const codeBlocks = node.querySelectorAll('pre code');
          codeBlocks.forEach((block) => {
            // Try to detect language from the code content
            const language = hljs.highlightAuto(block.textContent).language;
            if (language) {
              block.className = `hljs language-${language}`;
            }
            hljs.highlightElement(block);
          });
        }
      });
    });
  });

  observer.observe(document.getElementById('chatBox'), {
    childList: true,
    subtree: true
  });

  // Initial highlight for existing code blocks
  document.querySelectorAll('pre code').forEach((block) => {
    const language = hljs.highlightAuto(block.textContent).language;
    if (language) {
      block.className = `hljs language-${language}`;
    }
    hljs.highlightElement(block);
  });
}); 
document.getElementById('createAnimationBtn').addEventListener('click', () => {
  window.location.href = '/create_animation';
});
// Voice button functionality
document.getElementById('voiceBtn').addEventListener('click', () => {
  window.location.href = '/voice_chat';
});
// Add to your existing event listeners
document.getElementById('createSharedChat').addEventListener('click', async () => {
  try {
    const response = await fetch('/create_shared_chat', { method: 'POST' });
    const data = await response.json();
    
    if (data.status === 'success') {
      window.location.href = `/shared_chat?id=${data.chat_id}`;
    }
  } catch (error) {
    console.error('Error creating shared chat:', error);
  }
});

function joinSharedChat() {
  const chatCode = document.getElementById("chatCodeInput").value.trim();

  if (chatCode !== "") {
    const url = `/shared_chat?code=${encodeURIComponent(chatCode)}`;
    window.location.href = url;
  } else {
    alert("Please enter a valid chat code.");
  }
}
