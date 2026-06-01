import { Routes, Route } from 'react-router-dom'
import Header from './components/Layout/Header'
import Footer from './components/Layout/Footer'
import Home from './pages/Home'
import Visualizer from './pages/Visualizer'
import PhotoVisualizer from './pages/PhotoVisualizer'
import Calculator from './pages/Calculator'
import Catalog from './pages/Catalog'

export default function App() {
  return (
    <>
      <Header />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/visualizer" element={<Visualizer />} />
          <Route path="/photo" element={<PhotoVisualizer />} />
          <Route path="/calculator" element={<Calculator />} />
          <Route path="/catalog" element={<Catalog />} />
        </Routes>
      </main>
      <Footer />
    </>
  )
}
