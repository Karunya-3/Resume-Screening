'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  CheckCircle,
  TrendingUp,
  AlertCircle,
  Target,
  Star,
  Lightbulb,
  Download,
  Share2,
  Building,
  MapPin,
  Clock,
  Mail,
  Phone,
  FileText,
  User
} from 'lucide-react';

interface OutputProps {
  atsScore: number;
  resumeText: string;
  fileName: string;
  features: {
    word_count: number;
    skill_count: number;
    experience_years: number;
    section_count: number;
    has_email: boolean;
    has_phone: boolean;
  };
  onClose: () => void;
  onRetry?: () => void;
}

interface JobRecommendation {
  id: string;
  title: string;
  company: string;
  location: string;
  match_score: number;
  required_skills: string[];
  missing_skills: string[];
  salary_range: string;
  experience_level: string;
  description?: string;
  job_type?: string;
  posted_date?: string;
}

// 🔥 Clean AI API Fetcher
const fetchAIRecommendations = async (
  resumeText: string,
  atsScore: number
): Promise<JobRecommendation[]> => {
  try {
    console.log('🔍 Fetching AI job recommendations...');
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        resume_text: resumeText,
        features: {}
      })
    });

    const data = await response.json();
    console.log('🤖 AI returned:', data);

    return data.job_recommendations || [];
  } catch (error) {
    console.error('❌ AI fetch error:', error);
    return [];
  }
};

// Extract skills from resume text
const extractSkillsFromText = (text: string): string[] => {
  const commonSkills = [
    'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
    'node.js', 'express', 'django', 'flask', 'spring', 'spring boot',
    'html', 'css', 'sass', 'bootstrap', 'tailwind',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
    'aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'terraform',
    'machine learning', 'ai', 'data science', 'tensorflow', 'pytorch',
    'git', 'github', 'rest api', 'graphql', 'nosql',
    'c++', 'c#', 'php', 'ruby', 'go', 'rust',
    'tableau', 'power bi', 'excel', 'analytics',
    'agile', 'scrum', 'devops', 'ci/cd'
  ];

  const foundSkills = commonSkills.filter(skill =>
    text.toLowerCase().includes(skill.toLowerCase())
  );

  return [...new Set(foundSkills)];
};

export function Output({
  atsScore,
  resumeText,
  fileName,
  features,
  onClose,
  onRetry
}: OutputProps) {
  const [jobRecommendations, setJobRecommendations] = useState<JobRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [usingAI, setUsingAI] = useState(false);

  useEffect(() => {
    const loadJobs = async () => {
      setLoading(true);
      const aiJobs = await fetchAIRecommendations(resumeText, atsScore);

      if (aiJobs.length > 0) {
        setUsingAI(true);
        setJobRecommendations(aiJobs);
      } else {
        setUsingAI(false);
        setJobRecommendations([]);
      }

      setLoading(false);
    };

    loadJobs();
  }, [atsScore, features, resumeText]);

  // ATS scoring visuals
  const wordCountStatus =
    features.word_count >= 300 && features.word_count <= 800
      ? 'optimal'
      : features.word_count < 300
      ? 'too short'
      : 'too long';

  const skillCountStatus = features.skill_count >= 5 ? 'good' : 'needs improvement';
  const sectionCountStatus = features.section_count >= 4 ? 'good' : 'could be better';

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreVariant = (score: number) => {
    if (score >= 80) return 'default';
    if (score >= 60) return 'secondary';
    return 'destructive';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 80) return <CheckCircle className="w-5 h-5 text-green-600" />;
    if (score >= 60) return <TrendingUp className="w-5 h-5 text-yellow-600" />;
    return <AlertCircle className="w-5 h-5 text-red-600" />;
  };

  const getScoreMessage = (score: number) => {
    if (score >= 80) return 'Excellent! Your resume is optimized for ATS.';
    if (score >= 60) return 'Good! You can improve it further.';
    return 'Your resume needs optimization for ATS systems.';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'optimal':
      case 'good':
        return 'text-green-600';
      case 'could be better':
        return 'text-yellow-600';
      default:
        return 'text-red-600';
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">

      {/* ATS Score */}
      <Card className="border-l-4 border-l-blue-500">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="w-6 h-6" />
            <span>ATS Score Analysis - {fileName}</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            <div className="text-center">
              <div className="relative inline-block">
                <div className="w-32 h-32 rounded-full border-8 border-gray-200 flex items-center justify-center">
                  <span className={`text-3xl font-bold ${getScoreColor(atsScore)}`}>
                    {atsScore.toFixed(2)}
                  </span>
                </div>
                <div className="absolute top-0 right-0">{getScoreIcon(atsScore)}</div>
              </div>
              <p className="mt-2 text-sm text-gray-600">Overall ATS Score</p>
              <Badge variant={getScoreVariant(atsScore)} className="mt-1">
                ATS Score
              </Badge>
            </div>

            <div className="md:col-span-2">
              <h4 className="font-semibold mb-4">{getScoreMessage(atsScore)}</h4>

              <div className="grid grid-cols-2 gap-4 text-sm">
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-blue-600" />
                      <span>Word Count</span>
                    </div>
                    <div className="text-right">
                      <span className="font-medium">{features.word_count}</span>
                      <br />
                      <span className={`text-xs ${getStatusColor(wordCountStatus)}`}>
                        ({wordCountStatus})
                      </span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Lightbulb className="w-4 h-4 text-green-600" />
                      <span>Skills Identified</span>
                    </div>
                    <div className="text-right">
                      <span className="font-medium">{features.skill_count}</span>
                      <br />
                      <span className={`text-xs ${getStatusColor(skillCountStatus)}`}>
                        ({skillCountStatus})
                      </span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4 text-purple-600" />
                      <span>Experience Years</span>
                    </div>
                    <span className="font-medium">{features.experience_years}</span>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Target className="w-4 h-4 text-orange-600" />
                      <span>Sections</span>
                    </div>
                    <div className="text-right">
                      <span className="font-medium">{features.section_count}</span>
                      <br />
                      <span className={`text-xs ${getStatusColor(sectionCountStatus)}`}>
                        ({sectionCountStatus})
                      </span>
                    </div>
                  </div>

                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Mail className="w-4 h-4 text-red-600" />
                      <span>Email</span>
                    </div>
                    <Badge variant={features.has_email ? 'default' : 'outline'}>
                      {features.has_email ? 'Present' : 'Missing'}
                    </Badge>
                  </div>

                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Phone className="w-4 h-4 text-green-600" />
                      <span>Phone</span>
                    </div>
                    <Badge variant={features.has_phone ? 'default' : 'outline'}>
                      {features.has_phone ? 'Present' : 'Missing'}
                    </Badge>
                  </div>
                </div>
              </div>

              <Progress value={atsScore} className="mt-4" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Job Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Star className="w-6 h-6" />
            <span>Job Recommendations</span>
            <Badge variant={usingAI ? 'default' : 'outline'} className="ml-2">
              {usingAI ? 'AI-Generated' : 'No Matches'}
            </Badge>
          </CardTitle>
        </CardHeader>

        <CardContent>
          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-2 text-gray-600">Analyzing job fit using AI...</p>
            </div>
          ) : jobRecommendations.length === 0 ? (
            <div className="text-center py-8">
              <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No job matches found.</p>
              <p className="text-sm text-gray-500 mt-2">Try adding more technical keywords.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {jobRecommendations.map((job, index) => (
                <div
                  key={job.id || index}
                  className="border rounded-lg p-4 hover:shadow-md transition-shadow"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <h4 className="font-semibold text-lg">{job.title}</h4>
                      <div className="flex items-center gap-4 mt-1 text-sm text-gray-600 flex-wrap">
                        <div className="flex items-center gap-1">
                          <Building className="w-4 h-4" />
                          <span>{job.company}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <MapPin className="w-4 h-4" />
                          <span>{job.location}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          <span>{job.experience_level}</span>
                        </div>
                        <div className="font-medium text-green-600">{job.salary_range}</div>
                      </div>
                    </div>
                    <Badge
                      variant={
                        job.match_score >= 80
                          ? 'default'
                          : job.match_score >= 60
                          ? 'secondary'
                          : 'outline'
                      }
                    >
                      {job.match_score.toFixed(1)}% Match
                    </Badge>
                  </div>

                  {job.description && (
                    <p className="text-sm text-gray-600 mb-3">{job.description}</p>
                  )}

                  <div className="mb-3">
                    <p className="text-sm font-medium mb-2">Required Skills:</p>
                    <div className="flex flex-wrap gap-1">
                      {job.required_skills.map((skill, skillIndex) => {
                        const userSkills = extractSkillsFromText(resumeText);
                        const hasSkill = userSkills.some(
                          userSkill =>
                            userSkill.toLowerCase().includes(skill.toLowerCase()) ||
                            skill.toLowerCase().includes(userSkill.toLowerCase())
                        );
                        return (
                          <Badge
                            key={skillIndex}
                            variant={hasSkill ? 'default' : 'outline'}
                            className={
                              hasSkill
                                ? 'bg-green-100 text-green-800 border-green-300'
                                : 'bg-gray-100 text-gray-800'
                            }
                          >
                            {skill} {hasSkill && '✓'}
                          </Badge>
                        );
                      })}
                    </div>
                  </div>

                  {job.missing_skills && job.missing_skills.length > 0 && (
                    <div className="bg-amber-50 p-3 rounded-lg border border-amber-200">
                      <p className="text-sm font-medium mb-2 text-amber-800 flex items-center gap-1">
                        <Lightbulb className="w-4 h-4" />
                        Skills to Improve Match:
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {job.missing_skills.map((skill, skillIndex) => (
                          <Badge
                            key={skillIndex}
                            variant="outline"
                            className="bg-amber-100 text-amber-800 border-amber-300"
                          >
                            {skill}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Footer Buttons */}
      <div className="flex space-x-4">
        <Button className="flex-1" variant="outline" onClick={onClose}>
          Back to Dashboard
        </Button>
        <Button className="flex-1">
          <Download className="w-4 h-4 mr-2" />
          Download Report
        </Button>
        <Button className="flex-1" variant="secondary">
          <Share2 className="w-4 h-4 mr-2" />
          Share Results
        </Button>
      </div>
    </div>
  );
}

