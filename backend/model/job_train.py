import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re
import warnings
import os
warnings.filterwarnings('ignore')

class MLJobRecommender:
    def __init__(self):
        self.job_df = None
        self.resume_df = None
        self.ats_model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.scaler = None
        
        # Recommendation system components
        self.similarity_vectorizer = None
        self.job_tfidf = None
        self.resume_tfidf = None
        self.is_trained = False
        
    def load_models_and_data(self):
        """Load trained models and datasets"""
        print("🔧 LOADING MODELS AND DATA")
        print("=" * 50)
        
        try:
            # Load trained ATS models
            with open('ats_model.pkl', 'rb') as f:
                self.ats_model = pickle.load(f)
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            print("✅ ATS models loaded successfully")
            print(f"   TF-IDF features: {self.tfidf_vectorizer.get_feature_names_out().shape[0]}")
            print(f"   Model expecting: {self.ats_model.n_features_in_} features")
            
            # Load datasets
            self.resume_df = pd.read_csv('processed_resumes_ats.csv')
            self.job_df = pd.read_csv('job_postings.csv')
            
            # Fix skills format if they're stored as strings
            self._fix_skills_format()
            
            print(f"✅ Data loaded: {len(self.resume_df)} resumes, {len(self.job_df)} jobs")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models/data: {e}")
            return False

    def _fix_skills_format(self):
        """Fix skills format if they're stored as strings instead of lists"""
        print("   Fixing skills format...")
        
        def parse_skills(skills):
            if isinstance(skills, list):
                return skills
            elif isinstance(skills, str):
                # Try to parse string representation of list
                if skills.startswith('[') and skills.endswith(']'):
                    try:
                        return eval(skills)
                    except:
                        pass
                # Split by common delimiters
                skills = re.split(r'[,;|]', skills)
                return [s.strip().strip("'\"") for s in skills if s.strip()]
            return ['Communication', 'Teamwork']
        
        self.job_df['required_skills'] = self.job_df['required_skills'].apply(parse_skills)
        print("   ✓ Skills format fixed")

    def train_recommendation_system(self):
        """Train the recommendation system and export as pickle"""
        print("\n🎯 TRAINING RECOMMENDATION SYSTEM")
        print("=" * 50)
        
        try:
            # Create feature matrices for jobs and resumes
            job_features = self._extract_job_features()
            resume_features = self._extract_resume_features()
            
            # Train similarity model
            self._train_similarity_model(job_features, resume_features)
            
            self.is_trained = True
            print("✅ Recommendation system trained successfully")
            
            # Export the trained recommender
            self.export_recommender()
            
            return True
            
        except Exception as e:
            print(f"❌ Error training recommendation system: {e}")
            return False

    def _extract_job_features(self):
        """Extract ML features from job data"""
        print("   Extracting job features...")
        
        job_features = []
        
        for _, job in self.job_df.iterrows():
            # Text features
            text_features = f"{job['title']} {job['category']} {job['description']}"
            
            # Skills features
            skills_text = ' '.join(job['required_skills']) if isinstance(job['required_skills'], list) else str(job['required_skills'])
            
            # Numerical features
            salary_mid = (job['min_salary'] + job['max_salary']) / 2
            experience_weight = self._experience_to_weight(job['experience_level'])
            
            # Combine all features
            combined_features = f"{text_features} {skills_text} {salary_mid} {experience_weight}"
            job_features.append(combined_features)
        
        print(f"   ✓ Extracted features from {len(job_features)} jobs")
        return job_features

    def _extract_resume_features(self):
        """Extract ML features from resume data"""
        print("   Extracting resume features...")
        
        resume_features = []
        
        for _, resume in self.resume_df.iterrows():
            # Use existing resume text and category
            category = resume.get('Category', 'Unknown')
            resume_text = resume.get('resume_text', '')
            
            combined_features = f"{category} {resume_text}"
            resume_features.append(combined_features)
        
        print(f"   ✓ Extracted features from {len(resume_features)} resumes")
        return resume_features

    def _experience_to_weight(self, experience_level):
        """Convert experience level to numerical weight"""
        weights = {
            'Entry-level': 1,
            'Mid-level': 2,
            'Senior': 3
        }
        return weights.get(experience_level, 2)

    def _train_similarity_model(self, job_features, resume_features):
        """Train the ML similarity model"""
        print("   Training similarity model...")
        
        # Combine all text for TF-IDF
        all_texts = job_features + resume_features
        
        # Create TF-IDF matrix
        self.similarity_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.similarity_vectorizer.fit_transform(all_texts)
        
        # Split back into jobs and resumes
        self.job_tfidf = self.tfidf_matrix[:len(job_features)]
        self.resume_tfidf = self.tfidf_matrix[len(job_features):]
        
        print(f"   ✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def export_recommender(self, filename="job_recommender.pkl"):
        """Export the trained recommender as a pickle file"""
        print(f"\n💾 EXPORTING RECOMMENDER AS PICKLE: {filename}")
        
        try:
            # Create a dictionary with all necessary components
            recommender_data = {
                'similarity_vectorizer': self.similarity_vectorizer,
                'job_tfidf': self.job_tfidf,
                'resume_tfidf': self.resume_tfidf,
                'job_df': self.job_df,
                'is_trained': self.is_trained,
                'metadata': {
                    'jobs_count': len(self.job_df),
                    'resumes_count': len(self.resume_df),
                    'training_date': pd.Timestamp.now(),
                    'tfidf_shape': self.tfidf_matrix.shape
                }
            }
            
            # Save using joblib (better for large objects)
            joblib.dump(recommender_data, filename, compress=3)
            
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
            print(f"✅ Recommender exported successfully: {filename}")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Jobs: {len(self.job_df):,}")
            print(f"   Resumes: {len(self.resume_df):,}")
            print(f"   Training date: {recommender_data['metadata']['training_date']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting recommender: {e}")
            return False

    def load_recommender(self, filename="job_recommender.pkl"):
        """Load a trained recommender from pickle file"""
        print(f"📂 LOADING TRAINED RECOMMENDER: {filename}")
        
        try:
            if not os.path.exists(filename):
                print(f"❌ File not found: {filename}")
                return False
            
            # Load using joblib
            recommender_data = joblib.load(filename)
            
            # Restore all components
            self.similarity_vectorizer = recommender_data['similarity_vectorizer']
            self.job_tfidf = recommender_data['job_tfidf']
            self.resume_tfidf = recommender_data['resume_tfidf']
            self.job_df = recommender_data['job_df']
            self.is_trained = recommender_data['is_trained']
            
            metadata = recommender_data['metadata']
            print("✅ Recommender loaded successfully!")
            print(f"   Jobs: {metadata['jobs_count']:,}")
            print(f"   Resumes: {metadata['resumes_count']:,}")
            print(f"   Training date: {metadata['training_date']}")
            print(f"   TF-IDF shape: {metadata['tfidf_shape']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading recommender: {e}")
            return False

    def recommend_jobs_for_resume(self, resume_text, top_n=10):
        """Get job recommendations for a specific resume using ML"""
        if not self.is_trained:
            print("❌ Recommender not trained. Please train first or load a trained model.")
            return []
        
        print(f"\n🎯 GETTING RECOMMENDATIONS (Top {top_n})")
        print("=" * 50)
        
        try:
            # Method 1: Content-based filtering using TF-IDF similarity
            content_based_recs = self._content_based_recommendation(resume_text, top_n)
            
            # Method 2: Category-based matching (skip if feature mismatch)
            category_based_recs = self._category_based_recommendation(resume_text, top_n)
            
            # Method 3: Skill-based matching
            skill_based_recs = self._skill_based_recommendation(resume_text, top_n)
            
            # Combine and rank recommendations
            final_recommendations = self._rank_and_combine_recommendations(
                content_based_recs, category_based_recs, skill_based_recs, top_n
            )
            
            return final_recommendations
            
        except Exception as e:
            print(f"❌ Error in recommendation: {e}")
            return []

    def _content_based_recommendation(self, resume_text, top_n):
        """Content-based filtering using TF-IDF cosine similarity"""
        # Transform resume text to TF-IDF
        resume_tfidf = self.similarity_vectorizer.transform([resume_text])
        
        # Calculate cosine similarity with all jobs
        similarities = cosine_similarity(resume_tfidf, self.job_tfidf).flatten()
        
        # Get top matches
        top_indices = similarities.argsort()[-top_n*2:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if len(recommendations) >= top_n:
                break
                
            job = self.job_df.iloc[idx]
            recommendations.append({
                'job_id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'score': round(similarities[idx], 3),
                'method': 'Content-Based',
                'required_skills': job['required_skills'][:8],  # Limit skills display
                'category': job['category'],
                'experience_level': job['experience_level'],
                'salary_range': f"${job['min_salary']:,.0f} - ${job['max_salary']:,.0f}"
            })
        
        return recommendations

    def _category_based_recommendation(self, resume_text, top_n):
        """Category-based matching using resume category prediction"""
        try:
            # Transform using the correct vectorizer
            resume_tfidf = self.tfidf_vectorizer.transform([resume_text])
            
            # Ensure feature dimension matches
            if resume_tfidf.shape[1] != self.ats_model.n_features_in_:
                print(f"   ⚠️ Feature mismatch: {resume_tfidf.shape[1]} vs {self.ats_model.n_features_in_}")
                return []
            
            predicted_category_encoded = self.ats_model.predict(resume_tfidf)[0]
            predicted_category = self.label_encoder.inverse_transform([predicted_category_encoded])[0]
            
            print(f"   Predicted Category: {predicted_category}")
            
            # Find jobs in same category
            category_jobs = self.job_df[
                self.job_df['category'].str.contains(predicted_category, case=False, na=False)
            ]
            
            if len(category_jobs) > 0:
                category_jobs = category_jobs.sample(n=min(top_n, len(category_jobs)))
                
                recommendations = []
                for _, job in category_jobs.iterrows():
                    recommendations.append({
                        'job_id': job['id'],
                        'title': job['title'],
                        'company': job['company'],
                        'location': job['location'],
                        'score': 0.8,
                        'method': 'Category-Match',
                        'required_skills': job['required_skills'][:8],
                        'category': job['category'],
                        'experience_level': job['experience_level'],
                        'salary_range': f"${job['min_salary']:,.0f} - ${job['max_salary']:,.0f}"
                    })
                
                return recommendations
            
        except Exception as e:
            print(f"   Category prediction skipped: {e}")
        
        return []

    def _skill_based_recommendation(self, resume_text, top_n):
        """Skill-based matching between resume and job requirements"""
        resume_skills = self._extract_skills_from_text(resume_text)
        
        recommendations = []
        skill_scores = []
        
        for idx, job in self.job_df.iterrows():
            job_skills = job['required_skills'] if isinstance(job['required_skills'], list) else []
            
            if job_skills and resume_skills:
                common_skills = set(job_skills) & set(resume_skills)
                skill_score = len(common_skills) / max(len(job_skills), 1)
                
                if skill_score > 0.1:
                    skill_scores.append((idx, skill_score))
        
        skill_scores.sort(key=lambda x: x[1], reverse=True)
        
        for idx, score in skill_scores[:top_n]:
            job = self.job_df.iloc[idx]
            recommendations.append({
                'job_id': job['id'],
                'title': job['title'],
                'company': job['company'],
                'location': job['location'],
                'score': round(score, 3),
                'method': 'Skill-Match',
                'required_skills': job['required_skills'][:8],
                'category': job['category'],
                'experience_level': job['experience_level'],
                'salary_range': f"${job['min_salary']:,.0f} - ${job['max_salary']:,.0f}"
            })
        
        return recommendations

    def _extract_skills_from_text(self, text):
        """Extract skills from text"""
        skills_list = [
            'Python', 'Java', 'JavaScript', 'SQL', 'React', 'AWS', 'Docker', 'Kubernetes',
            'Machine Learning', 'Data Analysis', 'TensorFlow', 'PyTorch', 'Node.js',
            'HTML', 'CSS', 'Git', 'REST API', 'MongoDB', 'PostgreSQL', 'MySQL',
            'Azure', 'GCP', 'Jenkins', 'Terraform', 'Spring Boot', 'Django', 'Flask'
        ]
        
        text_lower = str(text).lower()
        found_skills = [skill for skill in skills_list if skill.lower() in text_lower]
        
        return found_skills

    def _rank_and_combine_recommendations(self, content_recs, category_recs, skill_recs, top_n):
        """Combine and rank recommendations from different methods"""
        all_recommendations = {}
        
        # Add content-based recommendations with weight
        for rec in content_recs:
            job_id = rec['job_id']
            if job_id not in all_recommendations:
                all_recommendations[job_id] = rec
                all_recommendations[job_id]['final_score'] = rec['score'] * 0.5
        
        # Add category-based recommendations with weight
        for rec in category_recs:
            job_id = rec['job_id']
            if job_id not in all_recommendations:
                all_recommendations[job_id] = rec
                all_recommendations[job_id]['final_score'] = rec['score'] * 0.3
            else:
                all_recommendations[job_id]['final_score'] += rec['score'] * 0.3
        
        # Add skill-based recommendations with weight
        for rec in skill_recs:
            job_id = rec['job_id']
            if job_id not in all_recommendations:
                all_recommendations[job_id] = rec
                all_recommendations[job_id]['final_score'] = rec['score'] * 0.2
            else:
                all_recommendations[job_id]['final_score'] += rec['score'] * 0.2
        
        # Convert to list and sort by final score
        final_list = list(all_recommendations.values())
        final_list.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_list[:top_n]

    def test_recommendation(self, resume_index=0, top_n=5):
        """Test the recommendation system with a sample resume"""
        if not self.is_trained:
            print("❌ Please train the recommender first")
            return
        
        sample_resume = self.resume_df.iloc[resume_index]
        print(f"\n🧪 TESTING WITH SAMPLE RESUME {resume_index}:")
        print(f"   Category: {sample_resume.get('Category', 'Unknown')}")
        print(f"   Text preview: {sample_resume['resume_text'][:200]}...")
        
        recommendations = self.recommend_jobs_for_resume(
            sample_resume['resume_text'], 
            top_n=top_n
        )
        
        # Display results
        print(f"\n🏆 TOP {top_n} JOB RECOMMENDATIONS:")
        print("=" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            skills_display = ', '.join(rec['required_skills'][:5]) if rec['required_skills'] else 'Not specified'
            print(f"\n{i}. {rec['title']}")
            print(f"   Company: {rec['company']}")
            print(f"   Location: {rec['location']}")
            print(f"   Category: {rec['category']}")
            print(f"   Experience: {rec['experience_level']}")
            print(f"   Salary: {rec['salary_range']}")
            print(f"   Match Score: {rec['final_score']:.3f} ({rec['method']})")
            print(f"   Key Skills: {skills_display}")


# Usage Examples
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 ML JOB RECOMMENDATION ENGINE WITH PICKLE EXPORT")
    print("=" * 60)
    
    # Option 1: Train and export new recommender
    recommender = MLJobRecommender()
    
    if recommender.load_models_and_data():
        # Train the recommendation system
        if recommender.train_recommendation_system():
            # Test with sample resume
            recommender.test_recommendation(resume_index=0, top_n=5)
    
    print("\n" + "=" * 60)
    print("💡 USAGE EXAMPLES:")
    print("=" * 60)
    print("""
    # Option 1: Train and export new recommender
    recommender = MLJobRecommender()
    recommender.load_models_and_data()
    recommender.train_recommendation_system()  # Exports as job_recommender.pkl
    
    # Option 2: Load existing trained recommender
    recommender = MLJobRecommender()
    recommender.load_recommender("job_recommender.pkl")
    
    # Get recommendations
    recommendations = recommender.recommend_jobs_for_resume(resume_text, top_n=10)
    """)