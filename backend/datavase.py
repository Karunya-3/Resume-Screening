import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        try:
            # Get MongoDB URI from environment
            mongodb_uri = os.getenv("MONGODB_URI")
            database_name = os.getenv("DATABASE_NAME", "resume_screening")
            
            if not mongodb_uri:
                logger.error("❌ MONGODB_URI not found in environment variables")
                return
            
            logger.info(f"🔗 Connecting to MongoDB Atlas...")
            logger.info(f"📁 Database: {database_name}")
            
            # Create client with timeout settings
            self.client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            
            logger.info("✅ Successfully connected to MongoDB Atlas!")
            
            # Initialize collections
            self._initialize_collections()
            
        except ServerSelectionTimeoutError as e:
            logger.error(f"❌ MongoDB connection timeout: {e}")
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to MongoDB: {e}")

    def _initialize_collections(self):
        """Initialize collections and indexes"""
        try:
            # Users collection
            users_collection = self.db.users
            users_collection.create_index("email", unique=True)
            users_collection.create_index("created_at")
            
            logger.info("✅ Database collections initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing collections: {e}")

    def get_collection(self, collection_name: str):
        if self.db is None:
            logger.error("❌ Database not connected")
            return None
        return self.db[collection_name]

    def close(self):
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")

    def is_connected(self):
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except:
            return False

# Global database instance
db = MongoDB()

def get_users_collection():
    return db.get_collection("users")