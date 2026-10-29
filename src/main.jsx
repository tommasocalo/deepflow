import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { useArchitectureStore } from './stores/architectureStore.js'
import { useTemplatesStore } from './stores/templatesStore.js'
import { useArchitectureHistory } from './stores/architectureHistory.js'
import { useTrainingStore } from './stores/trainingStore.js'
import * as validation from './validation/index.js'
import * as codegen from './codegen/index.js'
import { eduDevHandles, exposeEduHandle } from './edu/bridge.js'
import { useLessonSession } from './edu/lessons/engine/session.js'
import { useEduUiStore } from './edu/shell/uiStore.js'
import * as eduEngine from './edu/lessons/engine/devHandle.js'

// Development-only: expose the singleton stores so Playwright probes can
// drive the same app state instead of creating a parallel module instance
// via dynamic import. See scripts/*-test.mjs.
if (import.meta.env.DEV) {
    exposeEduHandle('session', useLessonSession);
    exposeEduHandle('progress', { getState: () => useLessonSession.getState().progress });
    exposeEduHandle('ui', useEduUiStore);
    exposeEduHandle('engine', eduEngine);
    window.__deepflow = {
        ...window.__deepflow,
        architectureStore: useArchitectureStore,
        architectureHistory: useArchitectureHistory,
        templatesStore: useTemplatesStore,
        trainingStore: useTrainingStore,
        validation,
        codegen,
        edu: eduDevHandles,
    };
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
