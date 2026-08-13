import { Navigate, Route, Routes } from 'react-router-dom'
import { Dashboard } from './pages/Dashboard'
import { NewAssessment } from './pages/NewAssessment'
import { ExplainableInsights } from './pages/ExplainableInsights'
import { ClinicalReport } from './pages/ClinicalReport'
import { PopulationAnalytics } from './pages/PopulationAnalytics'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/assessment/new" element={<NewAssessment />} />
      <Route path="/insights" element={<ExplainableInsights />} />
      <Route path="/reports" element={<ClinicalReport />} />
      <Route path="/analytics" element={<PopulationAnalytics />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
