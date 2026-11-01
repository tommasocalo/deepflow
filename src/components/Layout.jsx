import React, { useEffect } from 'react';
import Navbar from './Navbar.jsx';
import ToolbarStrip from './ToolbarStrip.jsx';
import ArchitectureWorkspace from './architecture/ArchitectureWorkspace.jsx';
import TrainingCanvas from './training/TrainingCanvas.jsx';
import TrainingPalette from './training/TrainingPalette.jsx';
import CodePanel from './CodePanel.jsx';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { useTrainingStore } from '../stores/trainingStore.js';
import { gpt2Storyboard } from '../storyboards/gpt2.js';
import { createDefaultModelSeeder } from '../storyboards/defaultModel.js';
import { startArchitectureSession } from '../stores/architectureHistory.js';
import { PRODUCT_NAME, PRODUCT_TAGLINE } from '../edu/config.js';
import LearnHome from '../edu/components/LearnHome.jsx';
import LessonPanel from '../edu/components/LessonPanel.jsx';
import DataWorkspace from '../edu/data/DataWorkspace.jsx';
import TrainerPanel from '../edu/trainer/components/TrainerPanel.jsx';
import { useLessonSession } from '../edu/lessons/engine/session.js';
import { installDeepLinks } from '../edu/shell/deepLink.js';
import { useEduUiStore } from '../edu/shell/uiStore.js';
import '../edu/shell/shell.css';

const seedDefaultModel = createDefaultModelSeeder({
    seed: () => startArchitectureSession(() => gpt2Storyboard({
        architectureStore: useArchitectureStore,
        templatesStore: useTemplatesStore,
        trainingStore: useTrainingStore,
    })),
});

const Layout = () => {
    const activeTab = useEduUiStore((s) => s.activeTab);
    const setTab = useEduUiStore((s) => s.setTab);
    const codeOpen = useEduUiStore((s) => s.codeOpen);
    const codeTab = useEduUiStore((s) => s.codeTab);
    const openCode = useEduUiStore((s) => s.openCode);
    const closeCode = useEduUiStore((s) => s.closeCode);
    const setCodeTab = useEduUiStore((s) => s.setCodeTab);
    const lessonMode = useLessonSession((s) => s.mode === 'lesson');

    // GPT-2 stays DeepFlow's first-start example and becomes the initial sandbox; deep links apply after it.
    useEffect(() => {
        seedDefaultModel();
        return installDeepLinks();
    }, []);
    useEffect(() => { document.title = `${PRODUCT_NAME} — ${PRODUCT_TAGLINE}`; }, []);

    return (
        <div className="df-app">
            <Navbar />
            <ToolbarStrip
                activeTab={activeTab}
                onChange={setTab}
                codeOpen={codeOpen}
                onShowCode={() => {
                    // Code toggles the panel, so it can be closed at any width.
                    if (codeOpen) { closeCode(); return; }
                    if (activeTab === 'learn') setTab('architecture');
                    openCode();
                }}
            />

            <div className={`df-app-main edu-main${lessonMode ? ' is-lesson' : ''}`}>
                <div className="edu-stage">
                    <div key={activeTab} className="df-tab-pane">
                        {activeTab === 'learn' && <LearnHome />}
                        {activeTab === 'architecture' && (
                            <ArchitectureWorkspace />
                        )}
                        {activeTab === 'data' && <DataWorkspace />}
                        {activeTab === 'training' && (
                            <div className="edu-training-split">
                                <div className="edu-training-canvas">
                                    <TrainingCanvas fitOnOpen={lessonMode} />
                                    <TrainingPalette />
                                </div>
                                <div className="edu-trainer-dock"><TrainerPanel /></div>
                            </div>
                        )}
                    </div>

                    <CodePanel open={codeOpen} onClose={closeCode} tab={codeTab} onTabChange={setCodeTab} />
                </div>
                {lessonMode && activeTab !== 'learn' && <LessonPanel />}
            </div>
        </div>
    );
};

export default Layout;
