import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ATSModelTrainer:
    """Train ML model to predict ATS scores"""
    
    def __init__(self):
        self.df = None
        self.tfidf = TfidfVectorizer(max_features=300, stop_words='english')
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load preprocessed resume data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        try:
            self.df = pd.read_csv('processed_resumes_ats.csv')
            print(f"\n✓ Loaded {len(self.df)} resumes")
            print(f"  Columns: {list(self.df.columns)}")
            
            # Check for required columns
            required_cols = ['resume_text', 'ats_score']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                print(f"✗ Missing required columns: {missing_cols}")
                return False
                
            return True
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("  Run preprocess.py first!")
            return False
    
    def prepare_features(self):
        """Prepare features for training"""
        print("\n" + "=" * 80)
        print("PREPARING FEATURES")
        print("=" * 80)
        
        # Clean data first
        print("\n1. Cleaning data...")
        self.df = self.df.dropna(subset=['resume_text'])
        self.df['resume_text'] = self.df['resume_text'].fillna('').astype(str)
        
        # TF-IDF features from resume text
        print("\n2. Creating TF-IDF features...")
        tfidf_features = self.tfidf.fit_transform(self.df['resume_text'])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        print(f"   ✓ Created {tfidf_features.shape[1]} TF-IDF features")
        
        # Select numeric features
        print("\n3. Selecting numeric features...")
        numeric_features = ['word_count', 'char_count', 'avg_word_length',
                           'skill_count', 'experience_years', 'has_email',
                           'has_phone', 'section_count']
        
        # Check which numeric features exist
        available_numeric = [f for f in numeric_features if f in self.df.columns]
        print(f"   ✓ Using {len(available_numeric)} numeric features: {available_numeric}")
        
        numeric_df = self.df[available_numeric].reset_index(drop=True)
        
        # Combine all features
        print("\n4. Combining features...")
        self.X = pd.concat([tfidf_df, numeric_df], axis=1)
        self.y = self.df['ats_score'].values
        
        print(f"   ✓ Total features: {self.X.shape[1]}")
        print(f"   ✓ Total samples: {self.X.shape[0]}")
        
        # Scale features
        print("\n5. Scaling features...")
        self.X = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.X.columns
        )
        print("   ✓ Features scaled")
    
    def split_data(self):
        """Split data into train and test sets"""
        print("\n" + "=" * 80)
        print("SPLITTING DATA")
        print("=" * 80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {len(self.X_train)} samples ({len(self.X_train)/len(self.X)*100:.1f}%)")
        print(f"Test set: {len(self.X_test)} samples ({len(self.X_test)/len(self.X)*100:.1f}%)")
    
    def train_model(self):
        """Train Random Forest model"""
        print("\n" + "=" * 80)
        print("TRAINING MODEL")
        print("=" * 80)
        
        print("\nModel: Random Forest Regressor")
        print("  • n_estimators: 100")
        print("  • max_depth: 15")
        print("  • min_samples_split: 5")
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        print("\nTraining...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Training complete!")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Metrics
        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        train_mae = mean_absolute_error(self.y_train, train_pred)
        test_mae = mean_absolute_error(self.y_test, test_pred)
        
        print("\nTraining Performance:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  RMSE: {train_rmse:.2f}")
        print(f"  MAE: {train_mae:.2f}")
        
        print("\nTest Performance:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.2f}")
        print(f"  MAE: {test_mae:.2f}")
        
        return train_pred, test_pred, test_r2, test_rmse
    
    def plot_results(self, train_pred, test_pred):
        """Plot model performance visualizations"""
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('ATS Score Prediction Model Performance', fontsize=14, fontweight='bold')
            
            # Plot 1: Actual vs Predicted (Test)
            axes[0, 0].scatter(self.y_test, test_pred, alpha=0.5)
            axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2)
            axes[0, 0].set_xlabel('Actual ATS Score')
            axes[0, 0].set_ylabel('Predicted ATS Score')
            axes[0, 0].set_title('Test Set: Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Residuals
            residuals = self.y_test - test_pred
            axes[0, 1].scatter(test_pred, residuals, alpha=0.5)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Predicted ATS Score')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residual Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Feature Importance (Top 15)
            feature_names = list(self.X.columns)
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-15:]
            
            axes[1, 0].barh(range(15), importances[indices])
            axes[1, 0].set_yticks(range(15))
            axes[1, 0].set_yticklabels([feature_names[i] for i in indices], fontsize=8)
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top 15 Feature Importance')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            
            # Plot 4: Score Distribution
            axes[1, 1].hist(self.y_test, bins=20, alpha=0.5, label='Actual', color='blue')
            axes[1, 1].hist(test_pred, bins=20, alpha=0.5, label='Predicted', color='orange')
            axes[1, 1].set_xlabel('ATS Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Score Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ats_model_performance.png', dpi=300, bbox_inches='tight')
            print("\n✓ Saved: ats_model_performance.png")
            plt.close()
            
        except Exception as e:
            print(f"✗ Error generating plots: {e}")
    
    def show_feature_importance(self):
        """Display top features"""
        print("\n" + "=" * 80)
        print("TOP 20 IMPORTANT FEATURES")
        print("=" * 80)
        
        feature_names = list(self.X.columns)
        importances = self.model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n", feature_importance.head(20).to_string(index=False))
    
    def save_model(self):
        """Save trained model and components"""
        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)
        
        try:
            # Save model
            with open('ats_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            print("\n✓ Saved: ats_model.pkl")
            
            # Save TF-IDF vectorizer
            with open('tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf, f)
            print("✓ Saved: tfidf_vectorizer.pkl")
            
            # Save scaler
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print("✓ Saved: scaler.pkl")
            
        except Exception as e:
            print(f"✗ Error saving models: {e}")
    
    def test_prediction(self):
        """Test prediction on sample resume"""
        print("\n" + "=" * 80)
        print("TESTING PREDICTION")
        print("=" * 80)
        
        try:
            # Get a sample from test set
            sample_idx = 0
            sample_features = self.X_test.iloc[sample_idx:sample_idx+1]
            actual_score = self.y_test[sample_idx]
            predicted_score = self.model.predict(sample_features)[0]
            
            print(f"\nSample Resume:")
            print(f"  Actual ATS Score: {actual_score:.2f}")
            print(f"  Predicted ATS Score: {predicted_score:.2f}")
            print(f"  Difference: {abs(actual_score - predicted_score):.2f}")
            
        except Exception as e:
            print(f"✗ Error testing prediction: {e}")


# Execute training pipeline
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" TRAIN.PY - ATS SCORE MODEL TRAINING")
    print("=" * 80)
    
    # Initialize trainer
    trainer = ATSModelTrainer()
    
    # Load preprocessed data
    if not trainer.load_data():
        exit(1)
    
    # Prepare features
    trainer.prepare_features()
    
    # Split data
    trainer.split_data()
    
    # Train model
    trainer.train_model()
    
    # Evaluate model
    train_pred, test_pred, r2, rmse = trainer.evaluate_model()
    
    # Show feature importance
    trainer.show_feature_importance()
    
    # Generate plots
    trainer.plot_results(train_pred, test_pred)
    
    # Save model
    trainer.save_model()
    
    # Test prediction
    trainer.test_prediction()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print(f"Model Performance: R² = {r2:.4f}, RMSE = {rmse:.2f}")
    print("=" * 80)