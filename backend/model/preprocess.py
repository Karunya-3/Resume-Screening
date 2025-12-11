import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

class ResumePreprocessor:
    """Preprocess resume datasets for ATS score training"""
    
    def __init__(self):
        self.resume_df = None
        
    def load_datasets(self):
        """Load and combine both resume datasets"""
        print("=" * 80)
        print("LOADING RESUME DATASETS")
        print("=" * 80)
        
        dfs = []
        
        # Load dataset 1: snehaanbhawal
        try:
            print("\n1. Loading Resume Dataset 1 (snehaanbhawal)...")
            df1 = pd.read_csv('Resume.csv')
            print(f"   ✓ Loaded {len(df1)} resumes")
            dfs.append(df1)
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        # Load dataset 2: saugataroyarghya
        try:
            print("\n2. Loading Resume Dataset 2 (saugataroyarghya)...")
            df2 = pd.read_csv('UpdatedResumeDataSet.csv')
            print(f"   ✓ Loaded {len(df2)} resumes")
            dfs.append(df2)
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        # Combine datasets
        if dfs:
            self.resume_df = pd.concat(dfs, ignore_index=True)
            print(f"\n✓ Combined total: {len(self.resume_df)} resumes")
        else:
            print("\n✗ No datasets loaded!")
            
    def clean_data(self):
        """Clean and preprocess resume data"""
        if self.resume_df is None:
            print("No data to clean!")
            return
        
        print("\n" + "=" * 80)
        print("CLEANING DATA")
        print("=" * 80)
        
        original = len(self.resume_df)
        
        # Remove duplicates
        self.resume_df.drop_duplicates(inplace=True)
        print(f"\n1. Removed {original - len(self.resume_df)} duplicates")
        
        # Handle missing values
        print("\n2. Handling missing values...")
        for col in self.resume_df.columns:
            if self.resume_df[col].isnull().sum() > 0:
                if self.resume_df[col].dtype == 'object':
                    self.resume_df[col].fillna('Not Specified', inplace=True)
                else:
                    self.resume_df[col].fillna(0, inplace=True)
        print(f"   ✓ All missing values handled")
        
        # Clean text columns
        print("\n3. Cleaning text...")
        text_cols = self.resume_df.select_dtypes(include=['object']).columns
        for col in text_cols:
            self.resume_df[col] = self.resume_df[col].apply(self.clean_text)
        print(f"   ✓ Cleaned {len(text_cols)} text columns")
        
        print(f"\n✓ Final count: {len(self.resume_df)} resumes")
    
    @staticmethod
    def clean_text(text):
        """Remove extra spaces and special characters"""
        if pd.isna(text) or not isinstance(text, str):
            return str(text)
        text = ' '.join(text.split())
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)
        return text.strip()
    
    def extract_features(self):
        """Extract features from resume text"""
        print("\n" + "=" * 80)
        print("EXTRACTING FEATURES")
        print("=" * 80)
        
        # Get resume text column (handle different column names)
        text_col = None
        for col in ['Resume_str', 'Resume', 'resume', 'text']:
            if col in self.resume_df.columns:
                text_col = col
                break
        
        if text_col is None:
            print("✗ No resume text column found!")
            return
        
        # Rename to standard 'resume_text'
        self.resume_df['resume_text'] = self.resume_df[text_col]
        
        # Extract word count
        print("\n1. Calculating word count...")
        self.resume_df['word_count'] = self.resume_df['resume_text'].apply(
            lambda x: len(str(x).split())
        )
        
        # Extract character count
        print("2. Calculating character count...")
        self.resume_df['char_count'] = self.resume_df['resume_text'].apply(
            lambda x: len(str(x))
        )
        
        # Extract average word length
        print("3. Calculating average word length...")
        self.resume_df['avg_word_length'] = (
            self.resume_df['char_count'] / (self.resume_df['word_count'] + 1)
        )
        
        # Extract skills count
        print("4. Counting skills...")
        self.resume_df['skill_count'] = self.resume_df['resume_text'].apply(
            self.count_skills
        )
        
        # Extract experience years
        print("5. Extracting experience years...")
        self.resume_df['experience_years'] = self.resume_df['resume_text'].apply(
            self.extract_experience
        )
        
        # Check for email
        print("6. Checking for email...")
        self.resume_df['has_email'] = self.resume_df['resume_text'].apply(
            lambda x: 1 if '@' in str(x) else 0
        )
        
        # Check for phone
        print("7. Checking for phone...")
        self.resume_df['has_phone'] = self.resume_df['resume_text'].apply(
            lambda x: 1 if re.search(r'\d{10}', str(x)) else 0
        )
        
        # Count sections (Education, Experience, Skills, etc.)
        print("8. Counting sections...")
        self.resume_df['section_count'] = self.resume_df['resume_text'].apply(
            self.count_sections
        )
        
        print("\n✓ Feature extraction complete!")
    
    @staticmethod
    def count_skills(text):
        """Count technical skills mentioned in resume"""
        skills = ['python', 'java', 'javascript', 'sql', 'c++', 'c#', 'ruby',
                 'php', 'swift', 'kotlin', 'react', 'angular', 'vue', 'node',
                 'django', 'flask', 'spring', 'html', 'css', 'aws', 'azure',
                 'docker', 'kubernetes', 'git', 'machine learning', 'data science',
                 'tensorflow', 'pytorch', 'excel', 'powerpoint', 'tableau']
        
        text_lower = str(text).lower()
        return sum(1 for skill in skills if skill in text_lower)
    
    @staticmethod
    def extract_experience(text):
        """Extract years of experience from text"""
        text_lower = str(text).lower()
        patterns = [r'(\d+)\+?\s*years?', r'(\d+)\s*-\s*\d+\s*years?']
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        return 0
    
    @staticmethod
    def count_sections(text):
        """Count resume sections like Education, Experience, Skills"""
        sections = ['education', 'experience', 'skills', 'projects', 
                   'certifications', 'achievements', 'summary', 'objective']
        text_lower = str(text).lower()
        return sum(1 for section in sections if section in text_lower)
    
    def create_ats_scores(self):
        """Generate synthetic ATS scores based on features"""
        print("\n" + "=" * 80)
        print("GENERATING ATS SCORES")
        print("=" * 80)
        
        print("\nScoring factors:")
        print("  • Resume length (300-800 words optimal)")
        print("  • Skills count (more is better)")
        print("  • Experience years")
        print("  • Contact info (email, phone)")
        print("  • Section structure")
        
        scores = []
        
        for idx, row in self.resume_df.iterrows():
            score = 50  # Base score
            
            # Word count scoring (optimal range)
            wc = row['word_count']
            if 300 <= wc <= 800:
                score += 15
            elif 200 <= wc < 300 or 800 < wc <= 1000:
                score += 10
            elif wc < 200:
                score += 5
            
            # Skills scoring
            score += min(row['skill_count'] * 3, 20)
            
            # Experience scoring
            score += min(row['experience_years'] * 2, 15)
            
            # Contact info scoring
            score += row['has_email'] * 5
            score += row['has_phone'] * 5
            
            # Section structure scoring
            score += min(row['section_count'] * 2, 10)
            
            # Add randomness for realism
            score += np.random.randint(-5, 6)
            
            # Ensure 0-100 range
            score = max(0, min(100, score))
            scores.append(score)
        
        self.resume_df['ats_score'] = scores
        
        print(f"\n✓ Generated ATS scores")
        print(f"  Mean: {np.mean(scores):.2f}")
        print(f"  Std: {np.std(scores):.2f}")
        print(f"  Min: {np.min(scores):.2f}")
        print(f"  Max: {np.max(scores):.2f}")
    
    def show_statistics(self):
        """Display dataset statistics"""
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        
        print(f"\nTotal resumes: {len(self.resume_df)}")
        print(f"Total features: {len(self.resume_df.columns)}")
        
        print("\nFeature statistics:")
        stats_cols = ['word_count', 'skill_count', 'experience_years', 
                     'section_count', 'ats_score']
        print(self.resume_df[stats_cols].describe())
        
        print("\nSample records:")
        print(self.resume_df[['word_count', 'skill_count', 'experience_years', 
                              'ats_score']].head())
    
    def save_processed_data(self):
        """Save processed dataset"""
        print("\n" + "=" * 80)
        print("SAVING DATA")
        print("=" * 80)
        
        self.resume_df.to_csv('processed_resumes_ats.csv', index=False)
        print("\n✓ Saved: processed_resumes_ats.csv")
        print(f"  Shape: {self.resume_df.shape}")
        print(f"  Columns: {list(self.resume_df.columns)}")


# Execute preprocessing
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" PREPROCESS.PY - ATS SCORE DATASET PREPARATION")
    print("=" * 80)
    
    # Create preprocessor
    preprocessor = ResumePreprocessor()
    
    # Load resume datasets
    preprocessor.load_datasets()
    
    # Clean data
    preprocessor.clean_data()
    
    # Extract features
    preprocessor.extract_features()
    
    # Generate ATS scores
    preprocessor.create_ats_scores()
    
    # Show statistics
    preprocessor.show_statistics()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE!")
    print("Next: Run train.py")
    print("=" * 80)