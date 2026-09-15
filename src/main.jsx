import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { useArchitectureStore } from './stores/architectureStore.js'
import { useTemplatesStore } from './stores/templatesStore.js'
import { useTrainingStore } from './stores/trainingStore.js'
import * as validation from './validation/index.js'
import * as codegen from './codegen/index.js'

// Development-only: expose the singleton stores so Playwright probes can
// drive the same app state instead of creating a parallel module instance
// via dynamic import. See scripts/*-test.mjs.
if (import.meta.env.DEV) {
    window.__deepflow = {
        architectureStore: useArchitectureStore,
        templatesStore: useTemplatesStore,
        trainingStore: useTrainingStore,
        validation,
        codegen,
    };
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
