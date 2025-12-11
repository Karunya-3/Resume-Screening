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

// 🔥 Clean AI API Fetcher with caching
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

// ENHANCED skill extraction with comprehensive pattern matching
const extractSkillsFromText = (text: string): string[] => {
  const textLower = text.toLowerCase();
  const foundSkills = new Set<string>();
  
  // Comprehensive skill database with variations and display names
  const skillDatabase: Record<string, { variations: string[], displayName: string }> = {
    // Programming Languages
    'python': { variations: ['python', 'py'], displayName: 'Python' },
    'java': { variations: ['java', 'j2ee', 'j2se'], displayName: 'Java' },
    'javascript': { variations: ['javascript', 'js', 'ecmascript'], displayName: 'JavaScript' },
    'typescript': { variations: ['typescript', 'ts'], displayName: 'TypeScript' },
    'c': { variations: ['\\bc\\b', '\\bc,', ', c,', ' c ', 'language c'], displayName: 'C' },
    'c++': { variations: ['c\\+\\+', 'cpp', 'cplusplus', 'c/c\\+\\+'], displayName: 'C++' },
    'c#': { variations: ['c#', 'csharp', 'c sharp'], displayName: 'C#' },
    'php': { variations: ['php'], displayName: 'PHP' },
    'ruby': { variations: ['ruby'], displayName: 'Ruby' },
    'go': { variations: ['\\bgo\\b', 'golang'], displayName: 'Go' },
    'rust': { variations: ['rust'], displayName: 'Rust' },
    'swift': { variations: ['swift'], displayName: 'Swift' },
    'kotlin': { variations: ['kotlin'], displayName: 'Kotlin' },
    'r': { variations: ['\\br\\b', 'r language', 'r programming'], displayName: 'R' },
    
    // Frontend Technologies
    'html': { variations: ['html', 'html5', 'html 5'], displayName: 'HTML' },
    'css': { variations: ['css', 'css3', 'css 3'], displayName: 'CSS' },
    'react': { variations: ['react', 'reactjs', 'react\\.js', 'react js'], displayName: 'React' },
    'angular': { variations: ['angular', 'angularjs'], displayName: 'Angular' },
    'vue': { variations: ['vue', 'vuejs', 'vue\\.js'], displayName: 'Vue' },
    'svelte': { variations: ['svelte'], displayName: 'Svelte' },
    'bootstrap': { variations: ['bootstrap'], displayName: 'Bootstrap' },
    'tailwind': { variations: ['tailwind', 'tailwindcss'], displayName: 'Tailwind CSS' },
    'sass': { variations: ['sass', 'scss'], displayName: 'SASS' },
    'jquery': { variations: ['jquery'], displayName: 'jQuery' },
    
    // Backend Frameworks
    'flask': { variations: ['flask'], displayName: 'Flask' },
    'django': { variations: ['django'], displayName: 'Django' },
    'node': { variations: ['node', 'nodejs', 'node\\.js', 'node js'], displayName: 'Node.js' },
    'express': { variations: ['express', 'expressjs', 'express\\.js'], displayName: 'Express' },
    'spring': { variations: ['spring', 'spring boot', 'springframework'], displayName: 'Spring' },
    'fastapi': { variations: ['fastapi', 'fast api'], displayName: 'FastAPI' },
    'rails': { variations: ['rails', 'ruby on rails'], displayName: 'Ruby on Rails' },
    'laravel': { variations: ['laravel'], displayName: 'Laravel' },
    'asp.net': { variations: ['asp\\.net', 'aspnet', 'asp net'], displayName: 'ASP.NET' },
    
    // Databases
    'sql': { variations: ['\\bsql\\b', 'structured query language'], displayName: 'SQL' },
    'mysql': { variations: ['mysql', 'my sql', 'my-sql'], displayName: 'MySQL' },
    'postgresql': { variations: ['postgresql', 'postgres', 'psql'], displayName: 'PostgreSQL' },
    'mongodb': { variations: ['mongodb', 'mongo', 'mongo db'], displayName: 'MongoDB' },
    'redis': { variations: ['redis'], displayName: 'Redis' },
    'oracle': { variations: ['oracle', 'oracle db'], displayName: 'Oracle' },
    'sqlite': { variations: ['sqlite', 'sqlite3'], displayName: 'SQLite' },
    'cassandra': { variations: ['cassandra'], displayName: 'Cassandra' },
    'dynamodb': { variations: ['dynamodb', 'dynamo db'], displayName: 'DynamoDB' },
    
    // Data Science & ML
    'numpy': { variations: ['numpy', 'np'], displayName: 'NumPy' },
    'pandas': { variations: ['pandas', 'pd'], displayName: 'Pandas' },
    'scikit-learn': { variations: ['scikit-learn', 'sklearn', 'scikit learn', 'scikitlearn'], displayName: 'Scikit-learn' },
    'tensorflow': { variations: ['tensorflow', 'tf'], displayName: 'TensorFlow' },
    'pytorch': { variations: ['pytorch', 'torch'], displayName: 'PyTorch' },
    'keras': { variations: ['keras'], displayName: 'Keras' },
    'machine learning': { variations: ['machine learning', 'ml', 'machine-learning'], displayName: 'Machine Learning' },
    'deep learning': { variations: ['deep learning', 'dl', 'deep-learning'], displayName: 'Deep Learning' },
    'ai': { variations: ['\\bai\\b', 'artificial intelligence'], displayName: 'AI' },
    'data analysis': { variations: ['data analysis', 'data analytics', 'data-analysis'], displayName: 'Data Analysis' },
    'data visualization': { variations: ['data visualization', 'data viz', 'dataviz'], displayName: 'Data Visualization' },
    'matplotlib': { variations: ['matplotlib'], displayName: 'Matplotlib' },
    'seaborn': { variations: ['seaborn'], displayName: 'Seaborn' },
    
    // Cloud & DevOps
    'aws': { variations: ['aws', 'amazon web services'], displayName: 'AWS' },
    'azure': { variations: ['azure', 'microsoft azure'], displayName: 'Azure' },
    'gcp': { variations: ['gcp', 'google cloud', 'google cloud platform'], displayName: 'Google Cloud' },
    'docker': { variations: ['docker'], displayName: 'Docker' },
    'kubernetes': { variations: ['kubernetes', 'k8s'], displayName: 'Kubernetes' },
    'jenkins': { variations: ['jenkins'], displayName: 'Jenkins' },
    'terraform': { variations: ['terraform'], displayName: 'Terraform' },
    'ansible': { variations: ['ansible'], displayName: 'Ansible' },
    'ci/cd': { variations: ['ci/cd', 'cicd', 'ci cd'], displayName: 'CI/CD' },
    
    // Tools & Version Control
    'git': { variations: ['\\bgit\\b', 'git version control'], displayName: 'Git' },
    'github': { variations: ['github'], displayName: 'GitHub' },
    'gitlab': { variations: ['gitlab'], displayName: 'GitLab' },
    'bitbucket': { variations: ['bitbucket'], displayName: 'Bitbucket' },
    'jira': { variations: ['jira'], displayName: 'JIRA' },
    'confluence': { variations: ['confluence'], displayName: 'Confluence' },
    'vs code': { variations: ['vs code', 'vscode', 'visual studio code'], displayName: 'VS Code' },
    'intellij': { variations: ['intellij', 'intellij idea'], displayName: 'IntelliJ' },
    'pycharm': { variations: ['pycharm'], displayName: 'PyCharm' },
    'eclipse': { variations: ['eclipse'], displayName: 'Eclipse' },
    'postman': { variations: ['postman'], displayName: 'Postman' },
    'figma': { variations: ['figma'], displayName: 'Figma' },
    'canva': { variations: ['canva'], displayName: 'Canva' },
    
    // Testing
    'jest': { variations: ['jest'], displayName: 'Jest' },
    'pytest': { variations: ['pytest'], displayName: 'Pytest' },
    'junit': { variations: ['junit'], displayName: 'JUnit' },
    'selenium': { variations: ['selenium'], displayName: 'Selenium' },
    'cypress': { variations: ['cypress'], displayName: 'Cypress' },
    
    // APIs & Protocols
    'rest': { variations: ['rest', 'rest api', 'restful', 'rest-api'], displayName: 'REST API' },
    'graphql': { variations: ['graphql', 'graph ql'], displayName: 'GraphQL' },
    'websocket': { variations: ['websocket', 'web socket', 'websockets'], displayName: 'WebSocket' },
    
    // Methodologies
    'agile': { variations: ['agile', 'agile methodology'], displayName: 'Agile' },
    'scrum': { variations: ['scrum'], displayName: 'Scrum' },
    'kanban': { variations: ['kanban'], displayName: 'Kanban' },
    
    // Soft Skills
    'leadership': { variations: ['leadership', 'team lead'], displayName: 'Leadership' },
    'communication': { variations: ['communication'], displayName: 'Communication' },
    'teamwork': { variations: ['teamwork', 'team work', 'collaboration'], displayName: 'Teamwork' },
    'problem solving': { variations: ['problem solving', 'problem-solving', 'problem resolution'], displayName: 'Problem Solving' },
    'management': { variations: ['management', 'project management'], displayName: 'Management' },
  };
  
  // Function to check if a skill is present using variations
  const checkSkill = (variations: string[], displayName: string) => {
    for (const variation of variations) {
      // Check if variation contains regex special characters
      if (variation.includes('\\b') || variation.includes('\\+') || variation.includes('\\.')) {
        try {
          const regex = new RegExp(variation, 'i');
          if (regex.test(text)) {
            foundSkills.add(displayName);
            return true;
          }
        } catch (e) {
          // If regex fails, try simple includes
          if (textLower.includes(variation.toLowerCase().replace(/\\/g, ''))) {
            foundSkills.add(displayName);
            return true;
          }
        }
      } else {
        // Simple substring match
        if (textLower.includes(variation.toLowerCase())) {
          foundSkills.add(displayName);
          return true;
        }
      }
    }
    return false;
  };
  
  // Check all skills in database
  Object.values(skillDatabase).forEach(skill => {
    checkSkill(skill.variations, skill.displayName);
  });
  
  // Enhanced comma-separated parsing
  // Pattern: "C, C++, Java, Python" or "HTML, CSS, JavaScript"
  const commaPattern = /\b([A-Za-z0-9+#.\/\-]+)\s*(?:,|and)\s*/gi;
  const lines = text.split('\n');
  
  lines.forEach(line => {
    const lineLower = line.toLowerCase();
    
    // Only parse lines that look like skill lists
    if (lineLower.includes('skill') || lineLower.includes('language') || 
        lineLower.includes('technolog') || lineLower.includes('tools') ||
        lineLower.includes('libraries') || lineLower.includes('framework')) {
      
      // Extract comma-separated items
      const matches = line.matchAll(commaPattern);
      for (const match of matches) {
        const potentialSkill = match[1].trim();
        const potentialSkillLower = potentialSkill.toLowerCase();
        
        // Check if this matches any skill in our database
        for (const [key, skill] of Object.entries(skillDatabase)) {
          if (skill.variations.some(v => {
            const cleanV = v.toLowerCase().replace(/\\/g, '').replace(/\\b/g, '');
            return potentialSkillLower === cleanV || 
                   potentialSkillLower.includes(cleanV) ||
                   cleanV.includes(potentialSkillLower);
          })) {
            foundSkills.add(skill.displayName);
            break;
          }
        }
      }
    }
  });
  
  // Special handling for "scikit learn" vs "scikit-learn"
  if (text.match(/scikit[\s-]learn/i)) {
    foundSkills.add('Scikit-learn');
  }
  
  const skillsArray = Array.from(foundSkills).sort();
  
  console.log('🎯 Enhanced extraction found skills:', skillsArray);
  console.log('📊 Total skills detected:', skillsArray.length);
  
  return skillsArray;
};

// 📊 Download Report as PDF-style HTML
const downloadReport = (
  atsScore: number,
  fileName: string,
  features: any,
  jobs: JobRecommendation[],
  resumeText: string
) => {
  const userSkills = extractSkillsFromText(resumeText);
  const reportDate = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>ATS Resume Analysis Report</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 40px 20px;
      line-height: 1.6;
      color: #333;
    }
    .header {
      text-align: center;
      margin-bottom: 40px;
      padding-bottom: 20px;
      border-bottom: 3px solid #3b82f6;
    }
    .header h1 {
      color: #1e40af;
      margin: 0;
      font-size: 32px;
    }
    .header .subtitle {
      color: #6b7280;
      margin-top: 10px;
    }
    .score-section {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 30px;
      border-radius: 12px;
      text-align: center;
      margin-bottom: 30px;
    }
    .score-large {
      font-size: 72px;
      font-weight: bold;
      margin: 10px 0;
    }
    .score-status {
      font-size: 18px;
      opacity: 0.9;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
      margin-bottom: 30px;
    }
    .metric-card {
      background: #f9fafb;
      padding: 20px;
      border-radius: 8px;
      border-left: 4px solid #3b82f6;
    }
    .metric-label {
      color: #6b7280;
      font-size: 14px;
      margin-bottom: 5px;
    }
    .metric-value {
      font-size: 24px;
      font-weight: bold;
      color: #1f2937;
    }
    .metric-status {
      font-size: 12px;
      color: #059669;
      margin-top: 5px;
    }
    .section-title {
      font-size: 24px;
      color: #1e40af;
      margin: 30px 0 15px 0;
      padding-bottom: 10px;
      border-bottom: 2px solid #e5e7eb;
    }
    .skills-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 30px;
    }
    .skill-badge {
      background: #dbeafe;
      color: #1e40af;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 500;
    }
    .job-card {
      background: white;
      border: 2px solid #e5e7eb;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 20px;
      page-break-inside: avoid;
    }
    .job-header {
      display: flex;
      justify-content: space-between;
      align-items: start;
      margin-bottom: 15px;
    }
    .job-title {
      font-size: 20px;
      font-weight: bold;
      color: #1f2937;
      margin: 0;
    }
    .match-score {
      background: #10b981;
      color: white;
      padding: 6px 16px;
      border-radius: 20px;
      font-weight: bold;
      font-size: 14px;
    }
    .match-score.medium {
      background: #f59e0b;
    }
    .match-score.low {
      background: #6b7280;
    }
    .job-meta {
      display: flex;
      gap: 20px;
      color: #6b7280;
      font-size: 14px;
      margin-bottom: 15px;
      flex-wrap: wrap;
    }
    .job-meta-item {
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .skills-section {
      margin-top: 15px;
    }
    .skills-section-title {
      font-weight: 600;
      color: #4b5563;
      margin-bottom: 8px;
      font-size: 14px;
    }
    .job-skill {
      display: inline-block;
      padding: 4px 10px;
      margin: 3px;
      border-radius: 4px;
      font-size: 13px;
    }
    .skill-match {
      background: #d1fae5;
      color: #065f46;
      border: 1px solid #6ee7b7;
    }
    .skill-missing {
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fca5a5;
    }
    .footer {
      margin-top: 50px;
      padding-top: 20px;
      border-top: 2px solid #e5e7eb;
      text-align: center;
      color: #6b7280;
      font-size: 14px;
    }
    @media print {
      body { padding: 20px; }
      .job-card { page-break-inside: avoid; }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>🎯 ATS Resume Analysis Report</h1>
    <div class="subtitle">
      <strong>Resume:</strong> ${fileName}<br>
      <strong>Generated:</strong> ${reportDate}
    </div>
  </div>

  <div class="score-section">
    <div class="score-status">Your ATS Compatibility Score</div>
    <div class="score-large">${atsScore}%</div>
    <div class="score-status">
      ${atsScore >= 80 ? '✅ Excellent - Highly compatible with ATS systems' :
      atsScore >= 60 ? '⚠️ Good - Some improvements recommended' :
        '❌ Needs Improvement - Optimize for better ATS performance'}
    </div>
  </div>

  <h2 class="section-title">📊 Resume Metrics</h2>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Word Count</div>
      <div class="metric-value">${features.word_count}</div>
      <div class="metric-status">
        ${features.word_count >= 300 && features.word_count <= 800 ? '✔ Optimal length' :
      features.word_count < 300 ? '⚠ Too short' : '⚠ Too long'}
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Technical Skills</div>
      <div class="metric-value">${features.skill_count}</div>
      <div class="metric-status">
        ${features.skill_count >= 5 ? '✔ Good variety' : '⚠ Add more skills'}
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Experience</div>
      <div class="metric-value">${features.experience_years} years</div>
      <div class="metric-status">✔ Detected</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Resume Sections</div>
      <div class="metric-value">${features.section_count}</div>
      <div class="metric-status">
        ${features.section_count >= 4 ? '✔ Well structured' : '⚠ Add more sections'}
      </div>
    </div>
  </div>

  <h2 class="section-title">💼 Your Technical Skills</h2>
  <div class="skills-list">
    ${userSkills.map(skill => `<span class="skill-badge">${skill}</span>`).join('')}
  </div>

  <h2 class="section-title">🎯 Recommended Job Matches (${jobs.length})</h2>
  ${jobs.map(job => `
    <div class="job-card">
      <div class="job-header">
        <h3 class="job-title">${job.title}</h3>
        <span class="match-score ${job.match_score >= 75 ? '' : job.match_score >= 60 ? 'medium' : 'low'}">
          ${job.match_score.toFixed(1)}% Match
        </span>
      </div>
      <div class="job-meta">
        <div class="job-meta-item">🏢 ${job.company}</div>
        <div class="job-meta-item">📍 ${job.location}</div>
        <div class="job-meta-item">💰 ${job.salary_range}</div>
        <div class="job-meta-item">⏱️ ${job.experience_level}</div>
      </div>
      ${job.description ? `<p style="color: #4b5563; margin: 15px 0;">${job.description}</p>` : ''}
      
      <div class="skills-section">
        <div class="skills-section-title">✅ Required Skills (Your Matches):</div>
        <div>
          ${job.required_skills.map(skill => {
        const hasSkill = userSkills.some(us =>
          us.toLowerCase().includes(skill.toLowerCase()) ||
          skill.toLowerCase().includes(us.toLowerCase())
        );
        return `<span class="job-skill ${hasSkill ? 'skill-match' : ''}">${skill}${hasSkill ? ' ✓' : ''}</span>`;
      }).join('')}
        </div>
      </div>

      ${job.missing_skills.length > 0 ? `
        <div class="skills-section">
          <div class="skills-section-title">📚 Skills to Learn (To Improve Match):</div>
          <div>
            ${job.missing_skills.map(skill =>
        `<span class="job-skill skill-missing">${skill}</span>`
      ).join('')}
          </div>
        </div>
      ` : ''}
    </div>
  `).join('')}

  <div class="footer">
    <p><strong>Resume ATS Analyzer</strong> - AI-Powered Career Intelligence</p>
    <p>This report was generated automatically based on industry-standard ATS criteria.</p>
  </div>
</body>
</html>
  `;

  // Create blob and download
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ATS_Report_${fileName.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.html`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
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

  // Fetch recommendations ONCE on mount
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
  }, []); // Empty dependency array = run only once

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

  // Extract user skills for display
  const userSkills = extractSkillsFromText(resumeText);

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
                      <span>Skills Found</span>
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
                      <span>Experience</span>
                    </div>
                    <span className="font-medium">{features.experience_years} years</span>
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

          {/* Display extracted skills */}
          {userSkills.length > 0 && (
            <div className="mt-6">
              <h4 className="font-semibold mb-2 text-sm flex items-center gap-2">
                <Lightbulb className="w-4 h-4" />
                Technical Skills Detected ({userSkills.length}):
              </h4>
              <div className="flex flex-wrap gap-2">
                {userSkills.map((skill, idx) => (
                  <Badge key={idx} variant="secondary" className="text-xs">
                    {skill}
                  </Badge>
                ))}
              </div>
            </div>
          )}
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
                      <div className="mt-2">
                        <Badge
                          variant={
                            job.match_score >= 80
                              ? 'default'
                              : job.match_score >= 60
                                ? 'secondary'
                                : 'outline'
                          }
                          className="text-sm font-medium"
                        >
                          Match: {job.match_score.toFixed(1)}%
                        </Badge>
                      </div>
                    </div>
                  </div>

                  {job.description && (
                    <p className="text-sm text-gray-600 mb-3">{job.description}</p>
                  )}

                  <div className="mb-3">
                    <p className="text-sm font-medium mb-2">Required Skills:</p>
                    <div className="flex flex-wrap gap-1">
                      {job.required_skills.map((skill, skillIndex) => {
                        const hasSkill = userSkills.some(userSkill => {
                          const userSkillLower = userSkill.toLowerCase().trim();
                          const jobSkillLower = skill.toLowerCase().trim();

                          if (userSkillLower === jobSkillLower) return true;
                          if (userSkillLower.includes(jobSkillLower) || jobSkillLower.includes(userSkillLower)) {
                            return true;
                          }

                          const commonVariations: Record<string, string[]> = {
                            'mysql': ['sql', 'mysql'],
                            'mongodb': ['mongo', 'mongodb'],
                            'flask': ['flask'],
                            'aws': ['aws'],
                            'azure': ['azure'],
                            'python': ['python', 'py'],
                          };

                          const userVariations = commonVariations[userSkillLower] || [userSkillLower];
                          const jobVariations = commonVariations[jobSkillLower] || [jobSkillLower];

                          return userVariations.some(uv =>
                            jobVariations.some(jv => uv === jv)
                          );
                        });

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

                  {(job.missing_skills && job.missing_skills.length > 0) && (
                    <div className="bg-amber-50 p-3 rounded-lg border border-amber-200">
                      <p className="text-sm font-medium mb-2 text-amber-800 flex items-center gap-1">
                        <Lightbulb className="w-4 h-4" />
                        Improve your match by learning:
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {job.missing_skills.map((skill, skillIndex) => {
                          const actuallyHasSkill = userSkills.some(userSkill => {
                            const userSkillLower = userSkill.toLowerCase().trim();
                            const missingSkillLower = skill.toLowerCase().trim();

                            if (userSkillLower === missingSkillLower) return true;
                            if (userSkillLower.includes(missingSkillLower) || missingSkillLower.includes(userSkillLower)) {
                              return true;
                            }

                            return false;
                          });

                          if (!actuallyHasSkill) {
                            return (
                              <Badge
                                key={skillIndex}
                                variant="outline"
                                className="bg-amber-100 text-amber-800 border-amber-300"
                              >
                                {skill}
                              </Badge>
                            );
                          }
                          return null;
                        })}
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
        <Button
          className="flex-1"
          onClick={() => downloadReport(atsScore, fileName, features, jobRecommendations, resumeText)}
          disabled={loading}
        >
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