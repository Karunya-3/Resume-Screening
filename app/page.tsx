"use client"

import { useState } from "react"
import { LoginPage } from "@/components/auth/login-page"
import { SignUpPage } from "@/components/auth/signup-page"
import { HomePage } from "@/components/home/home-page"
import { Dashboard } from "@/components/dashboard/dashboard"

export default function Page() {
  const [currentPage, setCurrentPage] = useState<"login" | "signup" | "home" | "dashboard">("login")
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  const handleLogin = () => {
    setIsAuthenticated(true)
    setCurrentPage("home")
  }

  const handleSignUp = () => {
    setIsAuthenticated(true)
    setCurrentPage("home")
  }

  const handleLogout = () => {
    setIsAuthenticated(false)
    setCurrentPage("login")
  }

  if (!isAuthenticated) {
    return (
      <>
        {currentPage === "login" && <LoginPage onLogin={handleLogin} onSignUpClick={() => setCurrentPage("signup")} />}
        {currentPage === "signup" && (
          <SignUpPage onSignUp={handleSignUp} onLoginClick={() => setCurrentPage("login")} />
        )}
      </>
    )
  }

  return (
    <>
      {currentPage === "home" && (
        <HomePage onDashboardClick={() => setCurrentPage("dashboard")} onLogout={handleLogout} />
      )}
      {currentPage === "dashboard" && <Dashboard onHomeClick={() => setCurrentPage("home")} onLogout={handleLogout} />}
    </>
  )
}
