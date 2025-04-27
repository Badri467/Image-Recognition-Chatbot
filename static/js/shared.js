// Global variables
let socket;
let currentChatId = null;
let username = "";
let shareCode = "";
let typingTimer = null;
let participants = {};

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', async () => {
  // Get username from server
  try {
    const response = await fetch('/get_user_info');
    const data = await response.json();
    username = data.username;
  } catch (error) {
    console.error('Error getting user info:', error);
    username = "Anonymous";
  }

  // Get chat ID from URL
  const urlParams = new URLSearchParams(window.location.search);
  const chatId = urlParams.get('id');
  const joinCode = urlParams.get('code');
  
  if (joinCode) {
    // If joining with a code
    await joinSharedChat(joinCode);
  } else if (chatId) {
    // If accessing with a chat ID
    await loadSharedChat(chatId);
  } else {
    // No ID or code, redirect to home
    window.location.href = '/';
  }
  
  // Theme toggle functionality
  initTheme();
  
  // Initialize Socket.IO connection
  initializeSocket();
  
  // Set up event listeners
  document.getElementById('sendButton').addEventListener('click', handleSend);
  document.getElementById('textQuery').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    
    // Send typing indicator
    if (!typingTimer) {
      socket.emit('typing', { 
        room: currentChatId,
        username: username
      });
    }
    
    // Clear previous timer
    clearTimeout(typingTimer);
    
    // Set new timer
    typingTimer = setTimeout(() => {
      socket.emit('stop_typing', { 
        room: currentChatId,
        username: username
      });
      typingTimer = null;
    }, 1000);
  });
  
  document.getElementById('copyShareCode').addEventListener('click', () => {
    navigator.clipboard.writeText(shareCode);
    showToast('Share code copied to clipboard!');
  });
  
  document.getElementById('copyShareLink').addEventListener('click', () => {
    const shareLink = `${window.location.protocol}//${window.location.host}/shared_chat?code=${shareCode}`;
    navigator.clipboard.writeText(shareLink);
    showToast('Share link copied to clipboard!');
  });
  
  document.getElementById('leaveChat').addEventListener('click', leaveChat);
  document.getElementById('returnToMainChat').addEventListener('click', () => {
    window.location.href = '/';
  });
  
  document.getElementById('toggleSidebar').addEventListener('click', () => {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('collapsed');
  });
  
  // Temperature slider functionality
  const temperatureSlider = document.getElementById('temperatureSlider');
  const temperatureValue = document.getElementById('temperatureValue');
  temperatureSlider.addEventListener('input', (e) => {
    temperatureValue.textContent = e.target.value;
  });
});

// Theme functionality
function initTheme() {
  const themeToggle = document.getElementById('themeToggle');
  const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
  
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
}


// Replace the initializeSocket function
function initializeSocket() {
    const serverUrl = window.location.protocol + '//' + window.location.hostname + 
                      (window.location.port ? ':' + window.location.port : '');
    
    socket = io(serverUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });
    
    const connectionStatus = document.getElementById('connectionStatus');
    
    socket.on('connect', () => {
      console.log('Connected to Socket.IO server');
      connectionStatus.textContent = 'Connected';
      connectionStatus.classList.add('visible', 'connected');
      setTimeout(() => {
        connectionStatus.classList.remove('visible');
      }, 3000);
      
      if (currentChatId) {
        joinChatRoom(currentChatId);
      }
    });
    
    socket.on('message', (data) => {
      // Handle incoming messages from other users
      if (data.user_id !== socket.id) {
        appendMessage('other-user', data.content, data.username);
      }
    });
    
    socket.on('bot_response', (data) => {
      appendMessage('bot', data.content + `\n\n---\nModel: ${data.model}`);
    });
    
    socket.on('user_joined', (data) => {
      // Add to participants
      participants[data.user_id] = {
        username: data.username,
        online: true
      };
      
      // Update participants list
      updateParticipantsList();
      
      // Show system message
      appendSystemMessage(`${data.username} joined the chat`);
    });
    
    socket.on('user_left', (data) => {
      if (participants[data.user_id]) {
        participants[data.user_id].online = false;
      }
      updateParticipantsList();
      appendSystemMessage(`${data.username} left the chat`);
    });
    
    socket.on('typing', (data) => {
      // Show typing indicator if it's not the current user
      if (data.user_id !== socket.id) {
        document.getElementById('typingIndicator').classList.remove('hidden');
        document.getElementById('typingIndicator').querySelector('span').textContent = 
          `${data.username} is typing...`;
      }
    });
    
    socket.on('stop_typing', () => {
      document.getElementById('typingIndicator').classList.add('hidden');
    });
    
    socket.on('disconnect', () => {
      console.log('Disconnected from Socket.IO server');
      connectionStatus.textContent = 'Disconnected';
      connectionStatus.classList.add('visible', 'error');
      connectionStatus.classList.remove('connected');
    });
    
    socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      connectionStatus.textContent = 'Connection Error';
      connectionStatus.classList.add('visible', 'error');
      connectionStatus.classList.remove('connected');
      showToast('Connection error: Please check your network', 'error');
    });
    
    socket.on('reconnect', (attemptNumber) => {
      console.log(`Reconnected after ${attemptNumber} attempts`);
      showToast('Reconnected to chat server');
      connectionStatus.textContent = 'Reconnected';
      connectionStatus.classList.add('visible', 'connected');
      connectionStatus.classList.remove('error');
      
      // Rejoin the room after reconnection
      if (currentChatId) {
        joinChatRoom(currentChatId);
      }
      
      setTimeout(() => {
        connectionStatus.classList.remove('visible');
      }, 3000);
    });
    
    socket.on('error', (error) => {
      console.error('Socket error:', error);
      showToast('Error: ' + error.message, 'error');
    });
  }
// Join a chat room
function joinChatRoom(roomId) {
  socket.emit('join', {
    room: roomId,
    username: username
  });
}

// Join a shared chat with code
async function joinSharedChat(code) {
  try {
    const response = await fetch(`/join_shared_chat/${code}`);
    const data = await response.json();
    
    if (!response.ok) {
      showToast(data.error || 'Failed to join chat', 'error');
      return;
    }
    
    // Set current chat ID
    currentChatId = data.chat_id;
    
    // Load chat data
    await loadSharedChat(currentChatId);
    
    // Update URL without refreshing
    window.history.replaceState(
      null, 
      'Shared Chat', 
      `/shared_chat?id=${currentChatId}`
    );
    
    showToast('Joined shared chat successfully');
  } catch (error) {
    console.error('Error joining shared chat:', error);
    showToast('Error joining shared chat', 'error');
  }
}

// Load shared chat data
async function loadSharedChat(chatId) {
  try {
    const response = await fetch(`/get_shared_chat/${chatId}`);
    const data = await response.json();
    
    if (!response.ok) {
      showToast(data.error || 'Failed to load chat', 'error');
      return;
    }
    
    // Set current chat ID and share code
    currentChatId = chatId;
    shareCode = data.share_code;
    
    // Update UI
    document.getElementById('chatTitle').textContent = data.title;
    document.getElementById('shareCodeDisplay').textContent = shareCode;
    document.getElementById('shareLinkDisplay').textContent = 
  `${window.location.protocol}//${window.location.host}/shared_chat?code=${shareCode}`;
    
    // Update participants
    data.participants.forEach(participant => {
      participants[participant.id] = {
        username: participant.username,
        online: false // Will be updated when they join the room
      };
    });
    updateParticipantsList();
    
    // Load messages
    const chatBox = document.getElementById('chatBox');
    chatBox.innerHTML = '';
    
    if (data.messages && data.messages.length > 0) {
      data.messages.forEach(message => {
        if (message.role === 'bot') {
          appendMessage('bot', message.content + `\n\n---\nModel: ${message.model}`);
        } else if (message.user_id === socket.id) {
          appendMessage('user', message.content);
        } else {
          appendMessage('other-user', message.content, message.username);
        }
      });
    } else {
      appendSystemMessage('Welcome to the shared chat session! You can now collaborate with others.');
    }
    
    // Join Socket.IO room
    joinChatRoom(currentChatId);
    
  } catch (error) {
    console.error('Error loading shared chat:', error);
    showToast('Error loading shared chat', 'error');
  }
}

// Update participants list
function updateParticipantsList() {
  const participantsList = document.getElementById('participantsList');
  participantsList.innerHTML = '';
  
  Object.entries(participants).forEach(([id, data]) => {
    const participantItem = document.createElement('div');
    participantItem.className = 'participant-item';
    
    const statusIndicator = document.createElement('div');
    statusIndicator.className = `status-indicator ${data.online ? 'online' : 'offline'}`;
    
    const username = document.createElement('span');
    username.textContent = data.username;
    
    participantItem.appendChild(statusIndicator);
    participantItem.appendChild(username);
    
    participantsList.appendChild(participantItem);
  });
}

// Leave the current chat
function leaveChat() {
  if (currentChatId) {
    socket.emit('leave', {
      room: currentChatId,
      username: username
    });
    
    // Redirect to home
    window.location.href = '/';
  }
}

// Handle sending a message
async function handleSend() {
  const queryInput = document.getElementById('textQuery');
  const query = queryInput.value.trim();
  const model = document.getElementById('modelSelect').value;
  const temperature = parseFloat(document.getElementById('temperatureSlider').value);

  if (!query || !currentChatId) return;

  // Clear input
  queryInput.value = '';
  
  // Clear typing timer and indicator
  clearTimeout(typingTimer);
  typingTimer = null;
  socket.emit('stop_typing', { 
    room: currentChatId,
    username: username
  });

  // Add message to chat
  appendMessage('user', query);
  
  // Send message to Socket.IO for other participants
  socket.emit('new_message', {
    room: currentChatId,
    message: query,
    username: username
  });
  
  try {
    // Send to server for AI processing
    const response = await fetch('/shared_query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: query,
        model: model,
        temperature: temperature,
        chat_id: currentChatId,
        username: username
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.response || 'Error processing query');
    }
    
    // We don't need to manually add the bot response here
    // The socket.io event will handle displaying the response
    
  } catch (error) {
    console.error('Error sending query:', error);
    appendSystemMessage(`Error: ${error.message}`, 'error');
  }
}

// Append message to chat
function appendMessage(type, content, sender = null) {
  const chatBox = document.getElementById('chatBox');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${type}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  
  if (type === 'other-user' && sender) {
    const senderName = document.createElement('div');
    senderName.className = 'sender-name';
    senderName.textContent = sender;
    bubble.appendChild(senderName);
  }
  
  if (type === 'bot') {
    // For bot messages, parse the markdown
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = content;
    bubble.innerHTML = tempDiv.innerHTML;
    
    // Apply syntax highlighting to code blocks
    bubble.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block);
    });
  } else {
    // For user messages, plain text is fine
    bubble.textContent = content;
  }
  
  messageDiv.appendChild(bubble);
  chatBox.appendChild(messageDiv);
  
  // Scroll to bottom
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Append system message
function appendSystemMessage(message, type = 'info') {
  const chatBox = document.getElementById('chatBox');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message system ${type}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = message;
  
  messageDiv.appendChild(bubble);
  chatBox.appendChild(messageDiv);
  
  // Scroll to bottom
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Show toast notification
function showToast(message, type = 'success') {
  // Create toast container if it doesn't exist
  let toastContainer = document.getElementById('toastContainer');
  
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toastContainer';
    document.body.appendChild(toastContainer);
  }
  
  // Create toast
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  
  // Add to container
  toastContainer.appendChild(toast);
  
  // Remove after 3 seconds
  setTimeout(() => {
    toast.classList.add('fadeOut');
    setTimeout(() => {
      toastContainer.removeChild(toast);
    }, 300);
  }, 3000);
}
// In shared.js
document.getElementById('usernameEdit').addEventListener('click', async () => {
    const newUsername = prompt('Enter new username (max 15 chars)');
    if (newUsername) {
      await fetch('/update_username', {
        method: 'POST',
        body: JSON.stringify({ newUsername }),
        headers: {'Content-Type': 'application/json'}
      });
      location.reload();
    }
  });