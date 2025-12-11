"use client"

import { useState, useEffect } from "react"
import { LoginPage } from "@/components/auth/login-page"
import { SignUpPage } from "@/components/auth/signup-page"
import { HomePage } from "@/components/home/home-page"
import { Dashboard } from "@/components/dashboard/dashboard"

export default function Page() {
  const [currentPage, setCurrentPage] = useState<"login" | "signup" | "home" | "dashboard">("login")
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState<any>(null)
  const [authToken, setAuthToken] = useState<string | null>(null)

  useEffect(() => {
    // Check if user is already logged in on component mount
    const token = localStorage.getItem('auth_token')
    const userData = localStorage.getItem('user')
    
    if (token && userData) {
      setAuthToken(token)
      setUser(JSON.parse(userData))
      setIsAuthenticated(true)
      setCurrentPage("home")
    }
  }, [])

  const handleLogin = (token: string, userData: any) => {
    setAuthToken(token)
    setUser(userData)
    setIsAuthenticated(true)
    setCurrentPage("home")
  }

  const handleSignUp = (token: string, userData: any) => {
    setAuthToken(token)
    setUser(userData)
    setIsAuthenticated(true)
    setCurrentPage("home")
  }

  const handleLogout = () => {
    // Clear localStorage
    localStorage.removeItem('auth_token')
    localStorage.removeItem('user')
    
    // Reset state
    setAuthToken(null)
    setUser(null)
    setIsAuthenticated(false)
    setCurrentPage("login")
  }

  if (!isAuthenticated) {
    return (
      <>
        {currentPage === "login" && (
          <LoginPage 
            onLogin={handleLogin} 
            onSignUpClick={() => setCurrentPage("signup")} 
          />
        )}
        {currentPage === "signup" && (
          <SignUpPage 
            onSignUp={handleSignUp} 
            onLoginClick={() => setCurrentPage("login")} 
          />
        )}
      </>
    )
  }

  return (
    <>
      {currentPage === "home" && (
        <HomePage 
          onDashboardClick={() => setCurrentPage("dashboard")} 
          onLogout={handleLogout} 
        />
      )}
      {currentPage === "dashboard" && (
        <Dashboard 
          {...({
            user,
            authToken,
            onHomeClick: () => setCurrentPage("home"),
            onLogout: handleLogout,
          } as any)}
        />
      )}
    </>
  )
}
