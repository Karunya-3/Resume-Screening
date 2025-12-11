from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
import os
import re
import json
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
import pickle
import pandas as pd
import numpy as np
import google.generativeai as genai
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
import certifi
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

#try:
    #client = AsyncIOMotorClient(
        #MONGODB_URI,
        #tls=True,
        #tlsCAFile=certifi.where()
    #)
    #db = client["resume"]
    #print("✓ MongoDB connected")
#except Exception as e:
    #print("❌ MongoDB connection failed:", e)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume ATS & Job Recommendation API", 
    version="3.0.0",
    description="Advanced Resume Screening with AI-powered Job Recommendations"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Security
security = HTTPBearer()

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "FuLtFyQ5gbWqPpvMnhgDnlryw12pvobNxpFwZgGb1G0")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Simple password hashing using SHA256
def get_password_hash(password: str) -> str:
    """Simple password hashing using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Simple password verification"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

# In-memory user storage (fallback when MongoDB is not available)
in_memory_users = {}

# MongoDB connection
class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        try:
            mongodb_uri = os.getenv("MONGODB_URI")
            database_name = os.getenv("DATABASE_NAME", "resume")

            if not mongodb_uri:
                logger.error("❌ MONGODB_URI not found in environment variables")
                return

            logger.info("🔗 Attempting to connect to MongoDB...")

            # Updated connection settings for SSL issues
            self.client = MongoClient(
                mongodb_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True
            )

            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[database_name]

            logger.info("✅ Successfully connected to MongoDB Atlas!")
            users_collection = self.db.users
            users_collection.create_index("email", unique=True)

        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            logger.info("💡 Using in-memory storage for development")



    def get_collection(self, collection_name: str):
        if self.db is None:
            return None
        return self.db[collection_name]

    def is_connected(self):
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except:
            return False

    def close(self):
        if self.client:
            self.client.close()

# Initialize database
db = MongoDB()

def get_users_collection():
    return db.get_collection("users")

# User storage functions with fallback
async def find_user_by_email(email: str):
    """Find user by email with MongoDB fallback to in-memory storage"""
    users_collection = get_users_collection()
    if users_collection is not None:
        try:
            return users_collection.find_one({"email": email})
        except Exception as e:
            logger.error(f"Database query failed: {e}")
    
    # Fallback to in-memory storage
    return in_memory_users.get(email)

async def create_user(user_data: dict):
    """Create user with MongoDB fallback to in-memory storage"""
    users_collection = get_users_collection()
    if users_collection is not None:
        try:
            result = users_collection.insert_one(user_data)
            user_data["_id"] = result.inserted_id
            logger.info(f"✅ User created in MongoDB: {user_data['email']}")
            return user_data
        except Exception as e:
            logger.error(f"Database insert failed: {e}")
    
    # Fallback to in-memory storage
    user_data["_id"] = str(len(in_memory_users) + 1)
    in_memory_users[user_data["email"]] = user_data
    logger.info(f"✅ User created in memory: {user_data['email']}")
    return user_data

# Simplified JWT functions
def create_access_token(data: dict):
    """Simple token creation using base64 encoding"""
    payload = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload.update({"exp": expire.timestamp()})
    
    payload_json = json.dumps(payload, default=str)
    payload_encoded = base64.urlsafe_b64encode(payload_json.encode()).decode()
    
    signature = hmac.new(
        SECRET_KEY.encode(),
        payload_encoded.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"{payload_encoded}.{signature}"

def verify_token(token: str):
    """Simple token verification"""
    try:
        if not token:
            return None
            
        parts = token.split('.')
        if len(parts) != 2:
            return None
            
        payload_encoded, signature = parts
        
        expected_signature = hmac.new(
            SECRET_KEY.encode(),
            payload_encoded.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            return None
            
        payload_json = base64.urlsafe_b64decode(payload_encoded).decode()
        payload = json.loads(payload_json)
        
        if datetime.utcnow().timestamp() > payload.get("exp", 0):
            return None
            
        return payload.get("sub")
        
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None

# ============================================================================
# ATS MODEL AND JOB RECOMMENDATION CODE
# ============================================================================

# Request/Response models for ATS
class ResumeRequest(BaseModel):
    resume_text: str
    features: Dict[str, Any] = {}

class RecommendationRequest(BaseModel):
    resume_text: str
    ats_score: float
    top_n: int = 5

class JobRecommendation(BaseModel):
    id: str
    title: str
    company: str
    location: str
    match_score: float
    required_skills: List[str]
    missing_skills: List[str]
    salary_range: str
    experience_level: str
    description: str

class PredictionResponse(BaseModel):
    ats_score: float
    confidence: float
    job_recommendations: List[JobRecommendation]
    features_used: Dict[str, Any]

class ATSModel:
    """Load and use trained ATS scoring model"""
    
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.scaler = None
        self.load_models()

    def load_models(self):
        """Load trained pickle files"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "model")
            
            model_path = os.path.join(model_dir, "ats_model.pkl")
            tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(tfidf_path, 'rb') as f:
                    self.tfidf = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✓ Loaded ATS models from: {model_dir}")
            else:
                logger.warning("⚠ ATS model files not found, using default scoring")
                
        except Exception as e:
            logger.error(f"✗ Error loading ATS models: {e}")

    def predict(self, resume_text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict ATS score using trained model or improved fallback scoring"""
        try:
            # If models are loaded, use them
            if self.model and self.tfidf and self.scaler:
                # Extract features
                word_count = len(resume_text.split())
                char_count = len(resume_text)
                avg_word_length = char_count / (word_count + 1)
                
                # Enhanced skills detection
                skills = ['python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 
                         'docker', 'kubernetes', 'machine learning', 'tensorflow', 'pytorch',
                         'angular', 'vue', 'spring', 'django', 'flask', 'mongodb', 'postgresql']
                skill_count = sum(1 for skill in skills if skill in resume_text.lower())
                
                # Experience years
                experience_match = re.search(r'(\d+)\+?\s*years?', resume_text.lower())
                experience_years = int(experience_match.group(1)) if experience_match else 0
                
                # Sections
                sections = ['education', 'experience', 'skills', 'projects', 'certifications']
                section_count = sum(1 for section in sections if section in resume_text.lower())
                
                # Contact info
                has_email = 1 if '@' in resume_text else 0
                has_phone = 1 if re.search(r'\d{10}', resume_text) else 0
                has_linkedin = 1 if 'linkedin' in resume_text.lower() else 0
                has_github = 1 if 'github' in resume_text.lower() else 0
                
                # Numeric features array
                numeric_features = [word_count, char_count, avg_word_length, skill_count, 
                                  experience_years, has_email, has_phone, section_count]
                
                # TF-IDF features
                tfidf_features = self.tfidf.transform([resume_text]).toarray()
                
                # Combine features
                all_features = np.concatenate([tfidf_features[0], numeric_features])
                
                # Scale and predict
                scaled_features = self.scaler.transform([all_features])
                base_ats_score = self.model.predict(scaled_features)[0]
                
                # Enhanced scoring with bonuses
                bonus = 0
                if skill_count > 5: bonus += 3
                if section_count >= 4: bonus += 2
                if has_linkedin: bonus += 1
                if has_github: bonus += 1
                if word_count > 300 and word_count < 1000: bonus += 2
                
                ats_score = min(100, max(0, base_ats_score + bonus))
                
                return {
                    'ats_score': round(ats_score, 2),
                    'confidence': round(0.85 + (skill_count * 0.01), 2),
                    'features_used': {
                        'word_count': word_count,
                        'skill_count': skill_count,
                        'experience_years': experience_years,
                        'section_count': section_count,
                        'has_email': bool(has_email),
                        'has_phone': bool(has_phone),
                        'bonus_points': bonus
                    }
                }
            else:
                # Enhanced fallback scoring if models not loaded
                return self.enhanced_fallback_scoring(resume_text, features)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self.enhanced_fallback_scoring(resume_text, features)

    def enhanced_fallback_scoring(self, resume_text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback ATS scoring with better logic"""
        word_count = len(resume_text.split())
        
        # Enhanced skill detection
        tech_skills = [
            'python', 'java', 'javascript', 'react', 'sql', 'html', 'css', 
            'mongodb', 'node', 'aws', 'docker', 'kubernetes', 'git', 'rest',
            'typescript', 'angular', 'vue', 'django', 'flask', 'spring'
        ]
        skill_count = sum(1 for skill in tech_skills if skill in resume_text.lower())
        
        # Enhanced scoring logic
        base_score = 50
        
        # Word count scoring (optimal: 300-800 words)
        if 300 <= word_count <= 800:
            base_score += 15
        elif 200 <= word_count < 300:
            base_score += 10
        elif 800 < word_count <= 1000:
            base_score += 8
        else:
            base_score += 5
        
        # Skills scoring
        base_score += min(skill_count * 4, 25)
        
        # Contact info scoring
        if '@' in resume_text: base_score += 5
        if re.search(r'\d{10}', resume_text): base_score += 5
        
        # Experience detection
        exp_patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?',
            r'(\d+)\s*years?\s*experience'
        ]
        
        has_experience = any(re.search(pattern, resume_text.lower()) for pattern in exp_patterns)
        if has_experience: base_score += 8
        
        # Section detection
        sections = ['education', 'experience', 'skills', 'projects', 'certifications']
        section_count = sum(1 for section in sections if section in resume_text.lower())
        base_score += min(section_count * 3, 12)
        
        ats_score = min(100, max(0, base_score))
        
        return {
            'ats_score': ats_score,
            'confidence': min(0.95, 0.7 + (skill_count * 0.03)),
            'features_used': {
                'word_count': word_count,
                'skill_count': skill_count,
                'has_email': '@' in resume_text,
                'has_phone': bool(re.search(r'\d{10}', resume_text)),
                'section_count': section_count,
                'has_experience': has_experience,
                'method': 'enhanced_fallback_scoring'
            }
        }

class GeminiJobRecommender:
    """Job Recommender using Gemini Flash AI"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-flash-latest')
                logger.info("✓ Gemini Flash AI initialized successfully")
            except Exception as e:
                logger.error(f"⚠ Gemini initialization failed: {e}")
                self.model = None
        else:
            logger.warning("⚠ No Gemini API key provided")
            self.model = None

    def recommend_jobs_gemini(self, resume_text: str, ats_score: float, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get job recommendations using Gemini AI"""
        if not self.model:
            return self.fallback_recommendations(resume_text, ats_score, top_k)
        
        try:
            # Limit resume text to avoid token limits
            limited_resume = resume_text[:1500]
            
            prompt = f"""
You are an expert technical career recommender.

Analyze the following resume (max 1500 chars):

RESUME:
{resume_text[:1500]}

ATS SCORE: {ats_score}/100

🎯 STRICT REQUIREMENTS:
- ONLY recommend *technical roles*.
- DO NOT recommend: Project Manager, Scrum Master, Consultant, Team Lead, Architect, or any non-coding roles.
- Roles MUST involve software development or data-related work.
- Skills to match: Python, JavaScript, React, HTML, CSS, MySQL, MongoDB, NumPy, Pandas, Scikit-learn, Java, C, C++, Flask.
- Roles must be one of the following types only:
    * Backend Developer (Python)
    * Full Stack Developer (React + Python)
    * Frontend Developer (React/JS)
    * Software Engineer
    * Data Analyst / ML Engineer
    * API Developer
    * Database Engineer

Return EXACT JSON (no markdown, no explanation):

{{
  "recommendations": [
    {{
      "id": "job_1",
      "title": "Job Title",
      "company": "Company Name",
      "location": "Location",
      "match_score": 85.5,
      "required_skills": ["Skill1", "Skill2", "Skill3"],
      "missing_skills": ["Skill4"],
      "salary_range": "$80,000 - $120,000",
      "experience_level": "Entry/Mid/Senior",
      "description": "Short technical job description"
    }}
  ]
}}

Rules:
- Match score must reflect skill overlap.
- Required skills must be technical only.
- Missing skills must be realistic and minimal.
- No managerial roles.
- No soft-skill-only recommendations.
"""

            logger.info(f"🤖 Sending request to Gemini with {len(limited_resume)} chars")
            
            response = self.model.generate_content(prompt)
            
            # Safer response text extraction
            response_text = ""
            try:
                if hasattr(response, 'text') and response.text:
                    response_text = response.text
                elif (hasattr(response, 'candidates') and response.candidates and 
                      len(response.candidates) > 0 and 
                      hasattr(response.candidates[0], 'content') and
                      hasattr(response.candidates[0].content, 'parts')):
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'):
                            response_text += part.text
                else:
                    logger.warning("⚠ No text content in Gemini response")
                    return self.fallback_recommendations(resume_text, ats_score, top_k)
            except Exception as e:
                logger.error(f"⚠ Error extracting Gemini response: {e}")
                return self.fallback_recommendations(resume_text, ats_score, top_k)

            logger.info(f"📄 Raw Gemini response: {response_text[:200]}...")

            # Better JSON extraction
            json_match = None
            try:
                patterns = [
                    r'\{[\s\S]*\}',
                    r'\{"recommendations"[\s\S]*\}',
                    r'\{.*"recommendations".*\}',
                ]
                
                for pattern in patterns:
                    json_match = re.search(pattern, response_text, re.DOTALL)
                    if json_match:
                        break
                        
                if not json_match:
                    logger.warning("⚠ Could not find JSON in Gemini response")
                    return self.fallback_recommendations(resume_text, ats_score, top_k)
                    
                json_str = json_match.group()
                logger.info(f"🎯 Extracted JSON: {json_str[:100]}...")
                
                recommendations_data = json.loads(json_str)
                recommendations = recommendations_data.get('recommendations', [])
                
                if recommendations:
                    logger.info(f"✅ Successfully parsed {len(recommendations)} recommendations from Gemini")
                    return recommendations
                else:
                    logger.warning("⚠ Gemini returned empty recommendations")
                    return self.fallback_recommendations(resume_text, ats_score, top_k)
                    
            except json.JSONDecodeError as e:
                logger.error(f"⚠ JSON parsing error: {e}")
                return self.fallback_recommendations(resume_text, ats_score, top_k)
                
        except Exception as e:
            logger.error(f"⚠ Gemini recommendation error: {e}")
            return self.fallback_recommendations(resume_text, ats_score, top_k)

    def fallback_recommendations(self, resume_text: str, ats_score: float, top_k: int = 5) -> List[Dict[str, Any]]:
        """Improved fallback job recommendations with better skill matching"""
        
        def extract_skills_intelligent(text):
            text_lower = text.lower()
            
            skill_variations = {
                'python': ['python', 'py'],
                'javascript': ['javascript', 'js', 'ecmascript'],
                'java': ['java', 'j2ee', 'j2se'],
                'react': ['react', 'reactjs', 'react.js'],
                'node': ['node', 'nodejs', 'node.js'],
                'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'oracle'],
                'html': ['html', 'html5'],
                'css': ['css', 'css3'],
                'mongodb': ['mongodb', 'mongo'],
                'aws': ['aws', 'amazon web services'],
                'docker': ['docker', 'container'],
                'kubernetes': ['kubernetes', 'k8s'],
                'git': ['git', 'github', 'gitlab'],
                'rest': ['rest', 'restful', 'rest api'],
                'typescript': ['typescript', 'ts'],
                'angular': ['angular', 'angularjs'],
                'vue': ['vue', 'vuejs'],
                'django': ['django'],
                'flask': ['flask'],
                'spring': ['spring', 'spring boot'],
                'c++': ['c++', 'cpp'],
                'c#': ['c#', 'csharp'],
                'php': ['php'],
                'ruby': ['ruby', 'rails'],
                'go': ['go', 'golang'],
                'rust': ['rust'],
                'machine learning': ['machine learning', 'ml', 'ai'],
                'tensorflow': ['tensorflow', 'tf'],
                'pytorch': ['pytorch'],
                'pandas': ['pandas'],
                'numpy': ['numpy'],
                'scikit': ['scikit', 'sklearn']
            }
            
            detected_skills = []
            for skill, variations in skill_variations.items():
                for variation in variations:
                    if variation in text_lower:
                        detected_skills.append(skill)
                        break
            
            return list(set(detected_skills))

        # Extract skills from resume
        detected_skills = extract_skills_intelligent(resume_text)
        logger.info(f"🎯 Detected skills in resume: {detected_skills}")
        
        # Enhanced job templates with realistic requirements
        job_templates = [
            {
                'id': 'FB_1', 
                'title': 'Full Stack Developer', 
                'company': 'Tech Solutions Inc',
                'location': 'Remote', 
                'required_skills': ['JavaScript', 'React', 'Node', 'HTML', 'CSS', 'SQL'],
                'salary_range': '$90,000 - $140,000', 
                'experience_level': 'Mid-level',
                'description': 'Develop and maintain web applications using modern technologies.'
            },
            {
                'id': 'FB_2', 
                'title': 'Python Developer', 
                'company': 'Data Corp',
                'location': 'San Francisco, CA', 
                'required_skills': ['Python', 'Django', 'SQL', 'REST', 'Git'],
                'salary_range': '$85,000 - $130,000', 
                'experience_level': 'Mid-level', 
                'description': 'Backend development with Python and web frameworks.'
            },
            {
                'id': 'FB_3', 
                'title': 'Software Engineer', 
                'company': 'Software Labs',
                'location': 'Austin, TX', 
                'required_skills': ['Java', 'Python', 'Git', 'SQL', 'REST'],
                'salary_range': '$95,000 - $145,000', 
                'experience_level': 'Mid-level',
                'description': 'Software development and engineering positions.'
            },
            {
                'id': 'FB_4', 
                'title': 'Data Analyst', 
                'company': 'Analytics Pro',
                'location': 'Chicago, IL', 
                'required_skills': ['Python', 'SQL', 'Machine Learning', 'Pandas', 'Statistics'],
                'salary_range': '$80,000 - $120,000', 
                'experience_level': 'Entry-level',
                'description': 'Analyze data and generate insights for business decisions.'
            },
            {
                'id': 'FB_5', 
                'title': 'Frontend Developer', 
                'company': 'Web Innovations',
                'location': 'New York, NY', 
                'required_skills': ['JavaScript', 'React', 'HTML', 'CSS', 'TypeScript'],
                'salary_range': '$85,000 - $135,000', 
                'experience_level': 'Mid-level',
                'description': 'Create responsive and interactive user interfaces.'
            }
        ]
        
        def calculate_match_score(job_skills, user_skills, ats_score):
            def normalize_skill(skill):
                return skill.lower().replace('.', '').replace(' ', '').replace('-', '')
            
            user_skills_normalized = [normalize_skill(skill) for skill in user_skills]
            job_skills_normalized = [normalize_skill(skill) for skill in job_skills]
            
            matches = 0
            missing_skills = []
            
            for job_skill in job_skills_normalized:
                found = False
                for user_skill in user_skills_normalized:
                    if job_skill in user_skill or user_skill in job_skill:
                        matches += 1
                        found = True
                        break
                if not found:
                    missing_skills.append(job_skills[job_skills_normalized.index(job_skill)])
            
            if len(job_skills) > 0:
                skill_ratio = matches / len(job_skills)
            else:
                skill_ratio = 0
                
            skill_score = skill_ratio * 70
            ats_contribution = (ats_score / 100) * 25
            experience_bonus = min(5, len(user_skills) * 0.5)
            
            total_score = skill_score + ats_contribution + experience_bonus
            final_score = max(40, min(95, total_score))
            
            return round(final_score, 1), missing_skills
        
        # Calculate matches for all jobs
        scored_jobs = []
        for job in job_templates:
            match_score, missing_skills = calculate_match_score(
                job['required_skills'], 
                detected_skills, 
                ats_score
            )
            
            scored_jobs.append({
                'id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'match_score': match_score,
                'required_skills': job['required_skills'],
                'missing_skills': missing_skills,
                'salary_range': job['salary_range'],
                'experience_level': job['experience_level'],
                'description': job['description']
            })
        
        # Sort by match score and return top_k
        sorted_jobs = sorted(scored_jobs, key=lambda x: x['match_score'], reverse=True)
        
        logger.info(f"🎯 Fallback generated {len(sorted_jobs)} jobs with scores: {[j['match_score'] for j in sorted_jobs[:top_k]]}")
        
        return sorted_jobs[:top_k]

# Initialize ATS and Gemini models
logger.info("Initializing ATS Model and Gemini Job Recommender...")
ats_model = ATSModel()
gemini_recommender = GeminiJobRecommender(os.getenv('GEMINI_API_KEY'))
logger.info("✓ ATS and Gemini initialization complete!\n")

# ============================================================================
# AUTHENTICATION MODELS AND DEPENDENCIES
# ============================================================================

# Pydantic Models for Authentication
class UserBase(BaseModel):
    name: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    email = verify_token(token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    
    user = await find_user_by_email(email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    return UserResponse(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        created_at=user["created_at"]
    )

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/signup", response_model=Token)
async def signup(user_data: UserCreate):
    try:
        # Check if user already exists
        existing_user = await find_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        user_dict = {
            "name": user_data.name,
            "email": user_data.email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow()
        }
        
        created_user = await create_user(user_dict)
        
        # Create access token
        access_token = create_access_token(data={"sub": user_data.email})
        
        user_response = UserResponse(
            id=str(created_user["_id"]),
            name=created_user["name"],
            email=created_user["email"],
            created_at=created_user["created_at"]
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in signup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account. Please try again."
        )

@app.post("/auth/login", response_model=Token)
async def login(login_data: UserLogin):
    try:
        print(f"🔐 Login attempt for: {login_data.email}")
        
        users_collection = get_users_collection()
        if users_collection is None:
            print("❌ Database connection failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database connection failed"
            )
        
        print("📊 Searching for user in database...")
        user = await find_user_by_email(login_data.email)
        print(f"👤 User found: {user is not None}")
        
        if not user:
            print("❌ User not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        print("🔑 Verifying password...")
        password_valid = verify_password(login_data.password, user["hashed_password"])
        print(f"✅ Password valid: {password_valid}")
        
        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        print("🎫 Creating access token...")
        access_token = create_access_token(data={"sub": user["email"]})
        
        user_response = UserResponse(
            id=str(user["_id"]),
            name=user["name"],
            email=user["email"],
            created_at=user["created_at"]
        )
        
        print(f"✅ Login successful for: {user['email']}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
    except HTTPException:
        print("🚨 HTTP Exception raised")
        raise
    except Exception as e:
        print(f"💥 Unexpected error in login: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_user)):
    return current_user

# ============================================================================
# ATS AND JOB RECOMMENDATION ENDPOINTS (PROTECTED)
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict_ats_score(
    request: ResumeRequest,
    #current_user: UserResponse = Depends(get_current_user)
):
    """Predict ATS score and get AI-powered job recommendations"""
    try:
        if not request.resume_text or len(request.resume_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Resume text is required")
        
        # Get ATS score prediction
        prediction = ats_model.predict(request.resume_text, request.features)
        
        # Get AI-powered job recommendations
        job_recommendations = gemini_recommender.recommend_jobs_gemini(
            request.resume_text,
            prediction['ats_score'],
            top_k=5
        )
        
        return PredictionResponse(
            ats_score=prediction['ats_score'],
            confidence=prediction['confidence'],
            job_recommendations=job_recommendations,
            features_used=prediction['features_used']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/recommend-jobs")
async def recommend_jobs_api(
    request: RecommendationRequest,
    #current_user: UserResponse = Depends(get_current_user)
):
    """API endpoint for job recommendations (used by React app)"""
    try:
        if not request.resume_text or len(request.resume_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Resume text is required")
        
        logger.info(f"📝 Processing AI job recommendations for user")
        
        job_recommendations = gemini_recommender.recommend_jobs_gemini(
            request.resume_text,
            request.ats_score,
            top_k=request.top_n
        )
        
        return {
            "recommendations": job_recommendations,
            "using_ai": gemini_recommender.model is not None,
            "message": f"Found {len(job_recommendations)} job recommendations"
        }
        
    except Exception as e:
        logger.error(f"Error in recommend-jobs API: {e}")
        raise HTTPException(status_code=500, detail=f"Job recommendation failed: {str(e)}")

# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Resume ATS & Job Recommendation API",
        "version": "3.0.0",
        "status": "running",
        "database_connected": db.is_connected(),
        "storage_mode": "MongoDB" if db.is_connected() else "In-Memory",
        "features": ["Authentication", "ATS Scoring", "AI Job Recommendations", "Gemini AI Integration"],
        "endpoints": {
            "auth": ["/auth/signup", "/auth/login", "/auth/me"],
            "ats": ["/predict", "/api/recommend-jobs"],
            "public": ["/", "/health", "/test"]
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_connected": db.is_connected(),
        "storage_mode": "MongoDB" if db.is_connected() else "In-Memory",
        "ats_model_loaded": ats_model.model is not None,
        "gemini_ai_loaded": gemini_recommender.model is not None,
        "users_count": len(in_memory_users),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/test")
async def test_endpoint():
    return {
        "message": "Backend is working!",
        "database_connected": db.is_connected(),
        "storage_mode": "MongoDB" if db.is_connected() else "In-Memory",
        "timestamp": datetime.utcnow().isoformat()
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Resume ATS API Server...")
    logger.info(f"📊 Database connected: {db.is_connected()}")
    logger.info(f"🤖 ATS Model loaded: {ats_model.model is not None}")
    logger.info(f"🤖 Gemini AI loaded: {gemini_recommender.model is not None}")
    if not db.is_connected():
        logger.info("💡 Using in-memory storage for development")

@app.on_event("shutdown")
async def shutdown_event():
    db.close()
    logger.info("🛑 Server shutting down...")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 80)
    print(" RESUME ATS & JOB RECOMMENDATION API")
    print("=" * 80)
    print(f"📊 Database Status: {'✅ Connected' if db.is_connected() else '❌ Disconnected'}")
    print(f"🤖 ATS Model: {'✅ Loaded' if ats_model.model is not None else '❌ Using Fallback'}")
    print(f"🤖 Gemini AI: {'✅ Loaded' if gemini_recommender.model is not None else '❌ Not Loaded'}")
    if not db.is_connected():
        print("💡 Using in-memory storage - authentication will work locally")
    print(f"🌐 Server starting on: http://localhost:8000")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")