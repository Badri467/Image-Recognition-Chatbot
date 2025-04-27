from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def init_db():
    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    db = client.chatbot_db

    # Create indexes
    db.chats.create_index([("user_id", 1), ("created_at", -1)])
    db.users.create_index("username", unique=True)
    db.fs.files.create_index("uploadDate")
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()