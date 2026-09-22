import React, { useEffect, useState } from 'react';
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

const seedDefaultModel = createDefaultModelSeeder({
    seed: () => startArchitectureSession(() => gpt2Storyboard({
        architectureStore: useArchitectureStore,
        templatesStore: useTemplatesStore,
        trainingStore: useTrainingStore,
    })),
});

const Layout = () => {
    const [activeTab, setActiveTab] = useState('architecture');
    const [codeOpen, setCodeOpen] = useState(false);

    useEffect(() => { seedDefaultModel(); }, []);

    return (
        <div className="df-app">
            <Navbar />
            <ToolbarStrip
                activeTab={activeTab}
                onChange={setActiveTab}
                onShowCode={() => setCodeOpen(true)}
            />

            <div className="df-app-main">
                <div key={activeTab} className="df-tab-pane">
                    {activeTab === 'architecture' && (
                        <ArchitectureWorkspace />
                    )}
                    {activeTab === 'training' && <TrainingCanvas />}
                    {activeTab === 'data' && (
                        <div className="w-full h-full flex items-center justify-center text-text-secondary text-sm">
                            <span className="border-2 border-black rounded-md px-3 py-1 bg-white shadow-medium">
                                Data canvas — coming in a later phase
                            </span>
                        </div>
                    )}
                </div>

                {activeTab === 'training' && <TrainingPalette />}

                <CodePanel open={codeOpen} onClose={() => setCodeOpen(false)} />
            </div>
        </div>
    );
};

export default Layout;
