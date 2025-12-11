"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Mail, Lock, Loader2, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface LoginPageProps {
  onLogin: (token: string, user: any) => void
  onSignUpClick: () => void
}

interface LoginResponse {
  access_token: string
  token_type: string
  user: {
    id: string
    name: string
    email: string
    created_at: string
  }
}

export function LoginPage({ onLogin, onSignUpClick }: LoginPageProps) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    
    if (!email || !password) {
      setError("Please fill in all fields")
      return
    }

    if (!validateEmail(email)) {
      setError("Please enter a valid email address")
      return
    }

    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email.trim(),
          password: password
        }),
      })

      const data = await response.json()

      if (response.ok) {
        // Store token in localStorage
        localStorage.setItem('auth_token', data.access_token)
        localStorage.setItem('user', JSON.stringify(data.user))
        
        // Call parent onLogin with token and user data
        onLogin(data.access_token, data.user)
      } else {
        setError(data.detail || 'Login failed. Please check your credentials.')
      }
    } catch (error) {
      console.error('Login error:', error)
      setError('Network error. Please check if the server is running.')
    } finally {
      setIsLoading(false)
    }
  }

  const validateEmail = (email: string) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return re.test(email)
  }

  const handleDemoLogin = async () => {
    setEmail("demo@example.com")
    setPassword("demopassword123")
    
    // Optional: Auto-submit demo credentials
    // await handleSubmit(new Event('submit') as any)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/20 via-background to-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <Card className="border-0 shadow-2xl backdrop-blur-sm bg-background/95">
          <CardHeader className="space-y-2 text-center">
            <div className="flex justify-center mb-4">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-xl">RS</span>
              </div>
            </div>
            <CardTitle className="text-3xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              Welcome Back
            </CardTitle>
            <CardDescription className="text-lg">
              Sign in to your resume screening account
            </CardDescription>
          </CardHeader>
          <CardContent>
            {error && (
              <Alert variant="destructive" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Email</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
                  <Input
                    type="email"
                    placeholder="you@example.com"
                    value={email}
                    onChange={(e) => {
                      setEmail(e.target.value)
                      setError("")
                    }}
                    className="pl-10 h-11"
                    disabled={isLoading}
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
                  <Input
                    type="password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => {
                      setPassword(e.target.value)
                      setError("")
                    }}
                    className="pl-10 h-11"
                    disabled={isLoading}
                    required
                    minLength={6}
                  />
                </div>
              </div>

              <Button
                type="submit"
                className="w-full h-11 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold transition-all duration-200"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Signing In...
                  </>
                ) : (
                  "Sign In"
                )}
              </Button>
            </form>

            {/* Demo Login Button (Optional) */}
            <div className="mt-4">
              <Button
                type="button"
                variant="outline"
                className="w-full h-11"
                onClick={handleDemoLogin}
                disabled={isLoading}
              >
                Fill Demo Credentials
              </Button>
            </div>

            <div className="mt-6 text-center">
              <p className="text-sm text-muted-foreground">
                Don't have an account?{" "}
                <button
                  onClick={onSignUpClick}
                  className="text-primary hover:text-primary/80 font-semibold transition-colors underline underline-offset-4"
                  disabled={isLoading}
                >
                  Sign up
                </button>
              </p>
            </div>

            {/* Server Status Indicator */}
            <div className="mt-4 text-center">
              <div className="flex items-center justify-center space-x-2 text-xs text-muted-foreground">
                <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`}></div>
                <span>Backend: {isLoading ? 'Connecting...' : 'Ready'}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
