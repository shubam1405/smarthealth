import { Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import PrivateRoute from './components/PrivateRoute'
import Sidebar from './components/Sidebar'
import ChatbotWidget from './components/ChatbotWidget'

// Pages
import Login from './pages/Login'
import Register from './pages/Register'
import PatientDashboard from './pages/PatientDashboard'
import DoctorDashboard from './pages/DoctorDashboard'
import DiabetesPrediction from './pages/DiabetesPrediction'
import HeartPrediction from './pages/HeartPrediction'
import XrayPrediction from './pages/XrayPrediction'

// Layout with sidebar + chatbot
function AppLayout({ children }) {
  return (
    <div className="layout">
      <Sidebar />
      <main className="main-content">{children}</main>
      <ChatbotWidget />
    </div>
  )
}

function AppRoutes() {
  const { user } = useAuth()

  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      {/* Root redirect */}
      <Route path="/" element={
        user
          ? <Navigate to={user.role === 'doctor' || user.role === 'admin' ? '/doctor' : '/patient'} replace />
          : <Navigate to="/login" replace />
      } />

      {/* Patient routes */}
      <Route path="/patient" element={
        <PrivateRoute roles={['patient']}>
          <AppLayout><PatientDashboard /></AppLayout>
        </PrivateRoute>
      } />
      <Route path="/patient/predict/diabetes" element={
        <PrivateRoute roles={['patient']}>
          <AppLayout><DiabetesPrediction /></AppLayout>
        </PrivateRoute>
      } />
      <Route path="/patient/predict/heart" element={
        <PrivateRoute roles={['patient']}>
          <AppLayout><HeartPrediction /></AppLayout>
        </PrivateRoute>
      } />
      <Route path="/patient/predict/xray" element={
        <PrivateRoute roles={['patient']}>
          <AppLayout><XrayPrediction /></AppLayout>
        </PrivateRoute>
      } />

      {/* Doctor routes */}
      <Route path="/doctor" element={
        <PrivateRoute roles={['doctor', 'admin']}>
          <AppLayout><DoctorDashboard /></AppLayout>
        </PrivateRoute>
      } />
      <Route path="/doctor/patients" element={
        <PrivateRoute roles={['doctor', 'admin']}>
          <AppLayout><DoctorDashboard /></AppLayout>
        </PrivateRoute>
      } />
      <Route path="/doctor/predict/diabetes" element={
        <PrivateRoute roles={['doctor', 'admin']}>
          <AppLayout><DiabetesPrediction /></AppLayout>
        </PrivateRoute>
      } />
      <Route path="/doctor/predict/heart" element={
        <PrivateRoute roles={['doctor', 'admin']}>
          <AppLayout><HeartPrediction /></AppLayout>
        </PrivateRoute>
      } />
      <Route path="/doctor/predict/xray" element={
        <PrivateRoute roles={['doctor', 'admin']}>
          <AppLayout><XrayPrediction /></AppLayout>
        </PrivateRoute>
      } />

      {/* Catch all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  )
}
