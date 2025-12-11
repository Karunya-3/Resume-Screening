const API_BASE_URL = 'http://localhost:8000';

export interface PredictionRequest {
  resume_text: string;
  features: {
    word_count: number;
    skill_count: number;
    experience_years: number;
    section_count: number;
    has_email: boolean;
    has_phone: boolean;
  };
}

export interface PredictionResponse {
  ats_score: number;
  confidence: number;
  job_recommendations: Array<{
    title: string;
    match_score: number;
    required_skills: string[];
    missing_skills: string[];
    salary_range: string;
    experience_level: string;
    company: string;
    location: string;
  }>;
  features_used: {
    word_count: number;
    skill_count: number;
    experience_years: number;
    section_count: number;
    has_email: boolean;
    has_phone: boolean;
  };
}

export async function predictATS(resumeData: PredictionRequest): Promise<PredictionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(resumeData),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Prediction API error:', error);
    throw error;
  }
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    return false;
  }
}
