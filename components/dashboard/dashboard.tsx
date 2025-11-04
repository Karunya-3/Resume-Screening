"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Home,
  LogOut,
  Upload,
  Search,
  Star,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  Trash2,
  FileText,
} from "lucide-react"

interface Resume {
  id: string
  name: string
  email: string
  position: string
  score: number | null
  status: "pending" | "approved" | "rejected"
  uploadDate: string
  file: File
}

interface DashboardProps {
  onHomeClick: () => void
  onLogout: () => void
}

export function Dashboard({ onHomeClick, onLogout }: DashboardProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [filterStatus, setFilterStatus] = useState<"all" | "pending" | "approved" | "rejected">("all")
  const [resumes, setResumes] = useState<Resume[]>([])

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files) return

    Array.from(files).forEach((file) => {
      if (file.type === "application/pdf" || file.type === "application/msword" || file.name.endsWith(".docx")) {
        const newResume: Resume = {
          id: Date.now().toString(),
          name: file.name.replace(/\.[^/.]+$/, ""),
          email: "",
          position: "",
          score: null,
          status: "pending",
          uploadDate: new Date().toISOString().split("T")[0],
          file: file,
        }
        setResumes((prev) => [newResume, ...prev])
      }
    })
    // Reset input
    event.target.value = ""
  }

  const handleDeleteResume = (id: string) => {
    setResumes((prev) => prev.filter((resume) => resume.id !== id))
  }

  const handleDownloadResume = (resume: Resume) => {
    const url = URL.createObjectURL(resume.file)
    const a = document.createElement("a")
    a.href = url
    a.download = resume.file.name
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const filteredResumes = resumes.filter((resume) => {
    const matchesSearch =
      resume.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      resume.email.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = filterStatus === "all" || resume.status === filterStatus
    return matchesSearch && matchesStatus
  })

  const stats = {
    total: resumes.length,
    approved: resumes.filter((r) => r.status === "approved").length,
    pending: resumes.filter((r) => r.status === "pending").length,
    rejected: resumes.filter((r) => r.status === "rejected").length,
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "approved":
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case "rejected":
        return <XCircle className="h-5 w-5 text-red-500" />
      case "pending":
        return <Clock className="h-5 w-5 text-yellow-500" />
      default:
        return null
    }
  }

  const getScoreColor = (score: number | null) => {
    if (score === null) return "text-muted-foreground"
    if (score >= 80) return "text-green-600"
    if (score >= 60) return "text-yellow-600"
    return "text-red-600"
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-primary-foreground font-bold">RS</span>
            </div>
            <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={onHomeClick}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
            >
              <Home className="h-5 w-5" />
              <span className="text-sm font-medium">Home</span>
            </button>
            <button
              onClick={onLogout}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
            >
              <LogOut className="h-5 w-5" />
              <span className="text-sm font-medium">Logout</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid md:grid-cols-4 gap-4 mb-8">
          <Card className="border border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Total Resumes</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-foreground">{stats.total}</div>
            </CardContent>
          </Card>

          <Card className="border border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Approved</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-600">{stats.approved}</div>
            </CardContent>
          </Card>

          <Card className="border border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Pending</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-yellow-600">{stats.pending}</div>
            </CardContent>
          </Card>

          <Card className="border border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Rejected</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-600">{stats.rejected}</div>
            </CardContent>
          </Card>
        </div>

        {/* Upload and Search Section */}
        <Card className="border border-border mb-8">
          <CardHeader>
            <CardTitle>Resume Screening</CardTitle>
            <CardDescription>Upload and manage candidate resumes</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col sm:flex-row gap-4">
              <label className="cursor-pointer">
                <Button className="bg-primary hover:bg-accent text-primary-foreground font-semibold w-full sm:w-auto">
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Resume
                </Button>
                <input type="file" multiple accept=".pdf,.doc,.docx" onChange={handleFileUpload} className="hidden" />
              </label>
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
                <Input
                  placeholder="Search by name or email..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            {/* Filter Buttons */}
            <div className="flex gap-2 flex-wrap">
              {(["all", "pending", "approved", "rejected"] as const).map((status) => (
                <button
                  key={status}
                  onClick={() => setFilterStatus(status)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    filterStatus === status
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground hover:bg-border"
                  }`}
                >
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Resumes Table */}
        <Card className="border border-border">
          <CardHeader>
            <CardTitle>Candidates ({filteredResumes.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {filteredResumes.length === 0 ? (
              <div className="text-center py-12">
                <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4 opacity-50" />
                <p className="text-muted-foreground mb-2">No resumes uploaded yet</p>
                <p className="text-sm text-muted-foreground">Upload your first resume to get started</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 font-semibold text-foreground">Name</th>
                      <th className="text-left py-3 px-4 font-semibold text-foreground">Score</th>
                      <th className="text-left py-3 px-4 font-semibold text-foreground">Status</th>
                      <th className="text-left py-3 px-4 font-semibold text-foreground">Date</th>
                      <th className="text-left py-3 px-4 font-semibold text-foreground">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredResumes.map((resume) => (
                      <tr key={resume.id} className="border-b border-border hover:bg-muted/50 transition-colors">
                        <td className="py-4 px-4">
                          <p className="font-medium text-foreground">{resume.name}</p>
                        </td>
                        <td className="py-4 px-4">
                          {resume.score !== null ? (
                            <div className="flex items-center gap-2">
                              <Star className="h-4 w-4 text-yellow-500 fill-yellow-500" />
                              <span className={`font-semibold ${getScoreColor(resume.score)}`}>{resume.score}%</span>
                            </div>
                          ) : (
                            <span className="text-sm text-muted-foreground">—</span>
                          )}
                        </td>
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(resume.status)}
                            <span className="text-sm font-medium text-foreground capitalize">{resume.status}</span>
                          </div>
                        </td>
                        <td className="py-4 px-4 text-sm text-muted-foreground">{resume.uploadDate}</td>
                        <td className="py-4 px-4">
                          <div className="flex gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleDownloadResume(resume)}
                              className="text-xs bg-transparent"
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleDeleteResume(resume.id)}
                              className="text-xs bg-transparent text-red-600 hover:text-red-700"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
