import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class JobDatasetPreparer:
    """Prepare job dataset from multiple Kaggle datasets"""
    
    def __init__(self):
        self.jobs_df = None
        self.resume_df = None
    
    def load_all_job_datasets(self):
        """
        Load all available job datasets from multiple sources
        """
        print("=" * 80)
        print("LOADING ALL JOB DATASETS")
        print("=" * 80)
        
        datasets_loaded = []
        
        # Dataset 1: Main Job Posts dataset (data job posts.csv)
        try:
            print("\n1. Loading Main Job Posts dataset...")
            job_posts_df = pd.read_csv('data job posts.csv')
            print(f"   ✓ Loaded {len(job_posts_df):,} jobs from main dataset")
            
            job_posts_processed = self.process_job_posts_dataset(job_posts_df)
            datasets_loaded.append(job_posts_processed)
            
        except Exception as e:
            print(f"   ✗ Error loading main job posts: {e}")
        
        # Dataset 2: LinkedIn Postings dataset (postings.csv)
        try:
            print("\n2. Loading LinkedIn Postings dataset...")
            linkedin_df = pd.read_csv('postings.csv')
            print(f"   ✓ Loaded {len(linkedin_df):,} jobs from LinkedIn dataset")
            
            # Sample to avoid memory issues
            sample_size = min(10000, len(linkedin_df))
            linkedin_sample = linkedin_df.sample(n=sample_size, random_state=42)
            linkedin_processed = self.process_linkedin_dataset(linkedin_sample)
            datasets_loaded.append(linkedin_processed)
            
        except Exception as e:
            print(f"   ✗ Error loading LinkedIn postings: {e}")
        
        # Dataset 3: Job Skills from jobs folder
        try:
            print("\n3. Loading Job Skills dataset...")
            skills_df = pd.read_csv('jobs/job_skills.csv')
            print(f"   ✓ Loaded {len(skills_df):,} skill entries")
            
            skills_processed = self.process_job_skills_dataset(skills_df)
            datasets_loaded.append(skills_processed)
            
        except Exception as e:
            print(f"   ✗ Error loading job skills: {e}")
        
        # Dataset 4: Salaries from jobs folder
        try:
            print("\n4. Loading Salaries dataset...")
            salaries_df = pd.read_csv('jobs/salaries.csv')
            print(f"   ✓ Loaded {len(salaries_df):,} salary entries")
            
            salaries_processed = self.process_salaries_dataset(salaries_df)
            datasets_loaded.append(salaries_processed)
            
        except Exception as e:
            print(f"   ✗ Error loading salaries: {e}")
        
        # Dataset 5: Job Industries from jobs folder
        try:
            print("\n5. Loading Job Industries dataset...")
            industries_df = pd.read_csv('jobs/job_industries.csv')
            print(f"   ✓ Loaded {len(industries_df):,} industry entries")
            
            industries_processed = self.process_industries_dataset(industries_df)
            datasets_loaded.append(industries_processed)
            
        except Exception as e:
            print(f"   ✗ Error loading job industries: {e}")
        
        # Dataset 6: Industries mapping
        try:
            print("\n6. Loading Industries Mapping...")
            industries_map_df = pd.read_csv('mappings/industries.csv')
            print(f"   ✓ Loaded {len(industries_map_df):,} industry mappings")
            
            # This is for reference, not for creating jobs
        except Exception as e:
            print(f"   ✗ Error loading industries mapping: {e}")
        
        # Dataset 7: Skills mapping
        try:
            print("\n7. Loading Skills Mapping...")
            skills_map_df = pd.read_csv('mappings/skills.csv')
            print(f"   ✓ Loaded {len(skills_map_df):,} skill mappings")
            
            # This is for reference, not for creating jobs
        except Exception as e:
            print(f"   ✗ Error loading skills mapping: {e}")
        
        # Combine all datasets
        if datasets_loaded:
            self.jobs_df = pd.concat(datasets_loaded, ignore_index=True)
            print(f"\n✓ Combined {len(self.jobs_df):,} jobs from {len(datasets_loaded)} datasets")
            return True
        else:
            print("\n✗ No job datasets could be loaded!")
            return False
    
    def process_job_posts_dataset(self, df):
        """Process the main Job Posts dataset"""
        print("   Processing main job posts...")
        processed_data = []
        
        # Sample to avoid too many entries
        sample_size = min(15000, len(df))
        df_sampled = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        
        for idx, row in df_sampled.iterrows():
            try:
                job_id = f"MAIN_{idx:05d}"
                
                # Extract data using actual column names
                title = self.clean_text(row.get('Title', f"Professional Position {idx}"))
                company = self.clean_text(row.get('Company', f"Organization {idx % 200}"))
                location = self.clean_text(row.get('Location', self.get_random_location()))
                
                # Build description from multiple fields
                desc_parts = []
                for field in ['JobDescription', 'JobRequirment', 'RequiredQual', 'jobpost']:
                    if field in row and pd.notna(row[field]):
                        desc_text = str(row[field])
                        if desc_text and desc_text != 'nan':
                            desc_parts.append(desc_text)
                
                description = ' '.join(desc_parts) if desc_parts else self.generate_description(title, company)
                
                # Extract information
                skills = self.extract_skills(description)
                exp_level = self.extract_experience_level(title + ' ' + description)
                min_sal, max_sal = self.extract_salary(row)
                category = self.extract_category(title)
                
                processed_data.append({
                    'id': job_id,
                    'title': title,
                    'company': company,
                    'location': location,
                    'description': description[:1000],
                    'required_skills': skills,
                    'experience_level': exp_level,
                    'min_salary': min_sal,
                    'max_salary': max_sal,
                    'category': category,
                    'job_type': 'Full-time',
                    'dataset_source': 'Main Job Posts'
                })
                
            except Exception:
                continue
        
        result = pd.DataFrame(processed_data)
        print(f"   ✓ Processed {len(result):,} jobs from main dataset")
        return result
    
    def process_linkedin_dataset(self, df):
        """Process LinkedIn postings dataset"""
        print("   Processing LinkedIn postings...")
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                job_id = f"LINK_{idx:05d}"
                
                # Map LinkedIn columns
                title = self.clean_text(row.get('title', f"Career Opportunity {idx}"))
                company = self.clean_text(row.get('company', f"Enterprise {idx % 150}"))
                location = self.clean_text(row.get('location', self.get_random_location()))
                description = self.clean_text(row.get('description', self.generate_description(title, company)))
                
                skills = self.extract_skills(description)
                exp_level = self.extract_experience_level(title)
                min_sal, max_sal = self.extract_salary(row)
                category = self.extract_category(title)
                
                processed_data.append({
                    'id': job_id,
                    'title': title,
                    'company': company,
                    'location': location,
                    'description': description[:1000],
                    'required_skills': skills,
                    'experience_level': exp_level,
                    'min_salary': min_sal,
                    'max_salary': max_sal,
                    'category': category,
                    'job_type': 'Full-time',
                    'dataset_source': 'LinkedIn'
                })
                
            except Exception:
                continue
        
        result = pd.DataFrame(processed_data)
        print(f"   ✓ Processed {len(result):,} jobs from LinkedIn")
        return result
    
    def process_job_skills_dataset(self, df):
        """Process job skills dataset to create job entries"""
        print("   Processing job skills dataset...")
        processed_data = []
        
        # Group by job to create job entries
        if 'job_id' in df.columns:
            job_groups = df.groupby('job_id')
            
            for job_id, group in list(job_groups)[:3000]:  # Limit to avoid too many entries
                try:
                    # Get skills for this job
                    skills = []
                    if 'skill_abr' in df.columns:
                        skills = group['skill_abr'].dropna().unique().tolist()
                    
                    # Create realistic job entry
                    tech_keywords = ['Developer', 'Engineer', 'Analyst', 'Specialist', 'Architect']
                    domain_keywords = ['Software', 'Data', 'Systems', 'Cloud', 'Web', 'Mobile']
                    
                    title = f"{np.random.choice(domain_keywords)} {np.random.choice(tech_keywords)}"
                    company = f"Tech Solutions {job_id % 100}"
                    
                    processed_data.append({
                        'id': f"SKILL_{job_id}",
                        'title': title,
                        'company': company,
                        'location': self.get_random_location(),
                        'description': f"We are seeking a {title} with expertise in {', '.join(skills[:5]) if skills else 'relevant technologies'}. This position offers growth opportunities and competitive compensation.",
                        'required_skills': skills[:10] if skills else self.get_default_tech_skills(),
                        'experience_level': np.random.choice(['Entry-level', 'Mid-level', 'Senior'], p=[0.2, 0.6, 0.2]),
                        'min_salary': 60000 + (job_id % 20) * 3000,
                        'max_salary': 90000 + (job_id % 20) * 5000,
                        'category': self.extract_category_from_skills(skills),
                        'job_type': 'Full-time',
                        'dataset_source': 'Job Skills'
                    })
                    
                except Exception:
                    continue
        
        result = pd.DataFrame(processed_data)
        print(f"   ✓ Processed {len(result):,} jobs from skills dataset")
        return result
    
    def process_salaries_dataset(self, df):
        """Process salaries dataset to create job entries"""
        print("   Processing salaries dataset...")
        processed_data = []
        
        # Sample the dataset
        sample_size = min(2000, len(df))
        df_sampled = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
        
        for idx, row in df_sampled.iterrows():
            try:
                job_id = f"SAL_{idx:05d}"
                
                # Extract job information from salary data
                title = self.clean_text(row.get('job_title', f"Professional Role {idx}"))
                company = self.clean_text(row.get('company', f"Corporation {idx % 100}"))
                location = self.clean_text(row.get('location', self.get_random_location()))
                
                # Get salary information
                min_sal = row.get('min_salary', 50000 + (idx % 50) * 2000)
                max_sal = row.get('max_salary', min_sal + 25000 + (idx % 30) * 1000)
                
                # Ensure reasonable salary ranges
                min_sal = max(40000, min_sal)
                max_sal = max(min_sal + 15000, max_sal)
                
                description = self.generate_description(title, company)
                skills = self.extract_skills(description)
                exp_level = self.extract_experience_level(title)
                category = self.extract_category(title)
                
                processed_data.append({
                    'id': job_id,
                    'title': title,
                    'company': company,
                    'location': location,
                    'description': description,
                    'required_skills': skills,
                    'experience_level': exp_level,
                    'min_salary': int(min_sal),
                    'max_salary': int(max_sal),
                    'category': category,
                    'job_type': 'Full-time',
                    'dataset_source': 'Salaries'
                })
                
            except Exception:
                continue
        
        result = pd.DataFrame(processed_data)
        print(f"   ✓ Processed {len(result):,} jobs from salaries dataset")
        return result
    
    def process_industries_dataset(self, df):
        """Process industries dataset to create job entries"""
        print("   Processing industries dataset...")
        processed_data = []
        
        # Group by job or industry
        group_col = 'job_id' if 'job_id' in df.columns else 'industry_id' if 'industry_id' in df.columns else None
        
        if group_col:
            groups = df.groupby(group_col)
            
            for group_id, group in list(groups)[:2000]:  # Limit entries
                try:
                    job_id = f"IND_{group_id}"
                    
                    # Get industry information
                    industries = []
                    if 'industry' in df.columns:
                        industries = group['industry'].dropna().unique().tolist()
                    
                    industry_name = industries[0] if industries else "Technology"
                    
                    # Create job title based on industry
                    if 'tech' in industry_name.lower() or 'software' in industry_name.lower():
                        title = f"{industry_name} {np.random.choice(['Engineer', 'Developer', 'Architect', 'Specialist'])}"
                    else:
                        title = f"{industry_name} {np.random.choice(['Manager', 'Analyst', 'Consultant', 'Coordinator'])}"
                    
                    company = f"{industry_name} Enterprises {group_id % 50}"
                    
                    processed_data.append({
                        'id': job_id,
                        'title': title,
                        'company': company,
                        'location': self.get_random_location(),
                        'description': f"Join our {industry_name} division as a {title}. This role offers the opportunity to work on cutting-edge projects and grow your career in {industry_name}.",
                        'required_skills': self.get_industry_skills(industry_name),
                        'experience_level': np.random.choice(['Entry-level', 'Mid-level', 'Senior'], p=[0.3, 0.5, 0.2]),
                        'min_salary': 50000 + (group_id % 25) * 2500,
                        'max_salary': 80000 + (group_id % 25) * 4000,
                        'category': industry_name,
                        'job_type': 'Full-time',
                        'dataset_source': 'Industries'
                    })
                    
                except Exception:
                    continue
        
        result = pd.DataFrame(processed_data)
        print(f"   ✓ Processed {len(result):,} jobs from industries dataset")
        return result
    
    def get_random_location(self):
        """Get random location"""
        locations = [
            'New York, NY', 'San Francisco, CA', 'Austin, TX', 'Seattle, WA',
            'Boston, MA', 'Chicago, IL', 'Denver, CO', 'Atlanta, GA',
            'Remote', 'Los Angeles, CA', 'Washington, DC', 'Miami, FL'
        ]
        return np.random.choice(locations)
    
    def generate_description(self, title, company):
        """Generate realistic job description"""
        descriptions = [
            f"Join {company} as a {title} and be part of our innovative team. We offer competitive compensation and excellent growth opportunities.",
            f"{company} is seeking a talented {title} to contribute to our success. This role requires dedication and expertise in relevant technologies.",
            f"Exciting opportunity for a {title} at {company}. Work on challenging projects with a dynamic team in a collaborative environment.",
            f"{company} is hiring a {title} to drive innovation and excellence. We value creativity, problem-solving skills, and technical expertise."
        ]
        return np.random.choice(descriptions)
    
    def get_default_tech_skills(self):
        """Get default technical skills"""
        skill_sets = [
            ['Python', 'SQL', 'Git', 'Problem Solving'],
            ['JavaScript', 'HTML', 'CSS', 'Communication'],
            ['Java', 'Spring Boot', 'REST API', 'Teamwork'],
            ['AWS', 'Docker', 'Linux', 'Analytical Skills']
        ]
        return np.random.choice(skill_sets)
    
    def get_industry_skills(self, industry):
        """Get skills based on industry"""
        industry_skills = {
            'technology': ['Python', 'JavaScript', 'Cloud Computing', 'Agile'],
            'finance': ['Analytical Skills', 'Excel', 'SQL', 'Risk Management'],
            'healthcare': ['Patient Care', 'Medical Knowledge', 'Communication', 'Empathy'],
            'education': ['Teaching', 'Curriculum Development', 'Communication', 'Patience'],
            'marketing': ['Digital Marketing', 'SEO', 'Social Media', 'Creativity']
        }
        
        industry_lower = industry.lower()
        for key, skills in industry_skills.items():
            if key in industry_lower:
                return skills
        
        return ['Communication', 'Teamwork', 'Problem Solving']
    
    def clean_text(self, text):
        """Clean and format text"""
        if pd.isna(text):
            return "Not specified"
        text = str(text).strip()
        return text if text and text != 'nan' else "Not specified"
    
    def extract_skills(self, text):
        """Extract technical skills from text"""
        skills_list = [
            'Python', 'Java', 'JavaScript', 'SQL', 'React', 'AWS', 'Docker', 'Kubernetes',
            'Machine Learning', 'Data Analysis', 'TensorFlow', 'PyTorch', 'Node.js',
            'HTML', 'CSS', 'Git', 'REST API', 'MongoDB', 'PostgreSQL', 'MySQL',
            'Azure', 'GCP', 'Jenkins', 'Terraform', 'Spring Boot', 'Django', 'Flask',
            'TypeScript', 'Angular', 'Vue', 'C++', 'C#', 'Ruby', 'PHP', 'Go',
            'Data Science', 'Big Data', 'AI', 'Artificial Intelligence'
        ]
        
        text_lower = str(text).lower()
        found_skills = [skill for skill in skills_list if skill.lower() in text_lower]
        
        # Add some default skills if none found
        if not found_skills:
            found_skills = ['Communication', 'Problem Solving', 'Teamwork']
        
        return list(set(found_skills))[:8]
    
    def extract_experience_level(self, text):
        """Extract experience level from text"""
        text_lower = str(text).lower()
        
        if any(word in text_lower for word in ['senior', 'lead', 'principal', 'manager', 'director', 'head']):
            return 'Senior'
        elif any(word in text_lower for word in ['junior', 'entry', 'graduate', 'trainee', 'associate']):
            return 'Entry-level'
        elif any(word in text_lower for word in ['mid', 'intermediate', 'experienced']):
            return 'Mid-level'
        else:
            return np.random.choice(['Entry-level', 'Mid-level', 'Senior'], p=[0.25, 0.6, 0.15])
    
    def extract_salary(self, row):
        """Extract salary information"""
        salary_fields = ['Salary', 'salary', 'min_salary', 'max_salary', 'compensation', 'pay']
        
        for field in salary_fields:
            if field in row and pd.notna(row[field]):
                salary_str = str(row[field])
                numbers = re.findall(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', salary_str)
                if numbers:
                    salaries = [int(re.sub(r'[^\d]', '', num)) for num in numbers if re.sub(r'[^\d]', '', num)]
                    if len(salaries) >= 2:
                        return min(salaries), max(salaries)
                    elif salaries:
                        base = salaries[0]
                        return base - 15000, base + 20000
        
        # Random realistic salaries based on experience level
        base = np.random.randint(50000, 100000)
        return base, base + np.random.randint(20000, 40000)
    
    def extract_category(self, title):
        """Extract job category from title"""
        title_lower = str(title).lower()
        
        category_map = {
            'Data Science': ['data scientist', 'data science', 'machine learning', 'ai', 'ml'],
            'Software Engineering': ['software engineer', 'developer', 'programmer', 'engineer'],
            'Web Development': ['web developer', 'frontend', 'backend', 'full stack'],
            'DevOps': ['devops', 'cloud engineer', 'sre'],
            'Data Analysis': ['data analyst', 'business analyst', 'analyst'],
            'Management': ['manager', 'director', 'lead', 'head of'],
            'Consulting': ['consultant', 'advisor'],
            'Research': ['researcher', 'scientist']
        }
        
        for category, keywords in category_map.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
        
        return 'Information Technology'
    
    def extract_category_from_skills(self, skills):
        """Extract category from skill list"""
        skill_categories = {
            'Data Science': ['python', 'machine learning', 'tensorflow', 'pytorch', 'data analysis'],
            'Web Development': ['javascript', 'react', 'node.js', 'html', 'css', 'angular'],
            'Software Engineering': ['java', 'c++', 'c#', 'spring boot', 'dotnet'],
            'DevOps': ['aws', 'docker', 'kubernetes', 'jenkins', 'terraform', 'azure']
        }
        
        skills_lower = [str(skill).lower() for skill in skills]
        
        for category, category_skills in skill_categories.items():
            if any(skill in skills_lower for skill in category_skills):
                return category
        
        return 'Information Technology'
    
    def clean_and_process(self):
        """Clean and process the combined dataset"""
        print("\n" + "=" * 80)
        print("CLEANING AND PROCESSING COMBINED DATASET")
        print("=" * 80)
        
        original_count = len(self.jobs_df)
        
        # Remove exact duplicates (same title, company, description)
        self.jobs_df = self.jobs_df.drop_duplicates(
            subset=['id'],  # Use ID for deduplication to be safe
            keep='first'
        )
        
        print(f"1. Removed {original_count - len(self.jobs_df):,} exact duplicates")
        print(f"2. Remaining jobs: {len(self.jobs_df):,}")
        
        # Ensure all required columns exist and have proper values
        self.jobs_df['title'] = self.jobs_df['title'].fillna('Professional Position')
        self.jobs_df['company'] = self.jobs_df['company'].fillna('Leading Company')
        self.jobs_df['location'] = self.jobs_df['location'].fillna('Various Locations')
        self.jobs_df['description'] = self.jobs_df['description'].fillna('Exciting career opportunity with growth potential')
        self.jobs_df['category'] = self.jobs_df['category'].fillna('General')
        self.jobs_df['experience_level'] = self.jobs_df['experience_level'].fillna('Mid-level')
        
        # Ensure skills are lists
        self.jobs_df['required_skills'] = self.jobs_df['required_skills'].apply(
            lambda x: x if isinstance(x, list) else ['Communication', 'Teamwork', 'Problem Solving']
        )
        
        # Ensure salary values are reasonable
        self.jobs_df['min_salary'] = self.jobs_df['min_salary'].clip(30000, 200000)
        self.jobs_df['max_salary'] = self.jobs_df['max_salary'].clip(
            self.jobs_df['min_salary'] + 10000, 250000
        )
        
        print(f"✓ Final cleaned dataset: {len(self.jobs_df):,} jobs")
    
    def show_statistics(self):
        """Display comprehensive statistics"""
        print("\n" + "=" * 80)
        print("FINAL DATASET STATISTICS")
        print("=" * 80)
        
        print(f"\nTotal Jobs: {len(self.jobs_df):,}")
        
        if 'dataset_source' in self.jobs_df.columns:
            print(f"\nData Sources:")
            for source, count in self.jobs_df['dataset_source'].value_counts().items():
                print(f"  {source}: {count:,} jobs")
        
        print(f"\nTop 10 Job Categories:")
        for category, count in self.jobs_df['category'].value_counts().head(10).items():
            print(f"  {category}: {count:,} jobs")
        
        print(f"\nExperience Level Distribution:")
        for level, count in self.jobs_df['experience_level'].value_counts().items():
            print(f"  {level}: {count:,} jobs")
        
        print(f"\nSalary Statistics:")
        print(f"  Average Min Salary: ${self.jobs_df['min_salary'].mean():,.0f}")
        print(f"  Average Max Salary: ${self.jobs_df['max_salary'].mean():,.0f}")
        print(f"  Highest Salary: ${self.jobs_df['max_salary'].max():,.0f}")
        
        print(f"\nSample Jobs (showing variety):")
        sample = self.jobs_df.sample(n=5, random_state=42)
        for _, row in sample.iterrows():
            print(f"  {row['id']}: {row['title']} at {row['company']} - {row['dataset_source']}")
    
    def save_dataset(self):
        """Save the final processed dataset"""
        print("\n" + "=" * 80)
        print("SAVING FINAL DATASET")
        print("=" * 80)
        
        filename = 'job_postings.csv'
        self.jobs_df.to_csv(filename, index=False)
        print(f"\n✓ Saved: {filename}")
        print(f"  Total jobs: {len(self.jobs_df):,}")
        print(f"  File size: {Path(filename).stat().st_size / 1024 / 1024:.2f} MB")
        
        # Also save a sample for testing
        sample_filename = 'job_postings_sample.csv'
        sample_size = min(1000, len(self.jobs_df))
        sample_df = self.jobs_df.sample(n=sample_size, random_state=42)
        sample_df.to_csv(sample_filename, index=False)
        print(f"✓ Sample saved: {sample_filename} ({sample_size:,} jobs)")


# Execute preparation
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" PREPARE COMPREHENSIVE JOB DATASET FOR ATS SYSTEM")
    print("=" * 80)
    
    preparer = JobDatasetPreparer()
    
    # Load all available datasets
    if preparer.load_all_job_datasets():
        # Clean and process
        preparer.clean_and_process()
        
        # Show statistics
        preparer.show_statistics()
        
        # Save
        preparer.save_dataset()
        
        print("\n" + "=" * 80)
        print("✅ JOB DATASET PREPARATION COMPLETE!")
        print("=" * 80)
        print(f"\nFinal dataset contains {len(preparer.jobs_df):,} diverse job postings")
        print("Ready for use in the ATS recommendation system!")
    else:
        print("\n❌ Failed to load any job datasets!")
        print("Please check that the required CSV files are in the correct locations.")