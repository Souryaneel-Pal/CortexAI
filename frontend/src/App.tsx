import { Navigate, Route, Routes } from 'react-router-dom'
import { AssessmentProvider } from './lib/assessmentContext'
import { SignIn } from './pages/SignIn'
import { Dashboard } from './pages/Dashboard'
import { NewAssessment } from './pages/NewAssessment'
import { Results } from './pages/Results'
import { ClinicalReport } from './pages/ClinicalReport'
import { PopulationAnalytics } from './pages/PopulationAnalytics'
import { Settings } from './pages/Settings'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = sessionStorage.getItem('auth_token')
  if (!token) {
    return <Navigate to="/" replace />
  }
  return <>{children}</>
}

function App() {
  // One provider above the router: the live assessment survives navigation
  // between Dashboard / Results / Reports / Analytics without refetching.
  return (
    <AssessmentProvider>
      <Routes>
        <Route path="/" element={<SignIn />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/assessment/new"
          element={
            <ProtectedRoute>
              <NewAssessment />
            </ProtectedRoute>
          }
        />
        <Route
          path="/results"
          element={
            <ProtectedRoute>
              <Results />
            </ProtectedRoute>
          }
        />
        <Route
          path="/reports"
          element={
            <ProtectedRoute>
              <ClinicalReport />
            </ProtectedRoute>
          }
        />
        <Route
          path="/analytics"
          element={
            <ProtectedRoute>
              <PopulationAnalytics />
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <Settings />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AssessmentProvider>
  )
}

export default App
