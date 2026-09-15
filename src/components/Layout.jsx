import React, { useState } from 'react';
import Navbar from './Navbar.jsx';
import ToolbarStrip from './ToolbarStrip.jsx';
import ArchitectureCanvas from './architecture/ArchitectureCanvas.jsx';
import BlockPalette from './architecture/BlockPalette.jsx';
import Breadcrumb from './architecture/Breadcrumb.jsx';
import SelectionToolbar from './architecture/SelectionToolbar.jsx';
import TemplateInspector from './architecture/TemplateInspector.jsx';
import TrainingCanvas from './training/TrainingCanvas.jsx';
import TrainingPalette from './training/TrainingPalette.jsx';
import CodePanel from './CodePanel.jsx';

const Layout = () => {
    const [activeTab, setActiveTab] = useState('architecture');
    const [codeOpen, setCodeOpen] = useState(false);

    return (
        <div
            style={{
                position: 'relative',
                width: '100%',
                height: '100%',
                backgroundColor: 'var(--color-bg)',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            <Navbar />
            <ToolbarStrip
                activeTab={activeTab}
                onChange={setActiveTab}
                onShowCode={() => setCodeOpen(true)}
            />

            <div style={{ position: 'relative', flex: 1, overflow: 'hidden' }}>
                <div
                    key={activeTab}
                    style={{ position: 'absolute', inset: 0, zIndex: 0, animation: 'tabFade 220ms ease' }}
                >
                    {activeTab === 'architecture' && <ArchitectureCanvas />}
                    {activeTab === 'training'     && <TrainingCanvas />}
                    {activeTab === 'data' && (
                        <div className="w-full h-full flex items-center justify-center text-text-secondary text-sm">
                            <span className="border-2 border-black rounded-md px-3 py-1 bg-white shadow-medium">
                                Data canvas — coming in a later phase
                            </span>
                        </div>
                    )}
                </div>

                {activeTab === 'architecture' && (
                    <>
                        <Breadcrumb />
                        <TemplateInspector />
                        <BlockPalette />
                        <SelectionToolbar />
                    </>
                )}

                {activeTab === 'training' && <TrainingPalette />}

                <CodePanel open={codeOpen} onClose={() => setCodeOpen(false)} />
            </div>

            <style>{`
                @keyframes tabFade {
                    from { opacity: 0; transform: translateY(4px); }
                    to   { opacity: 1; transform: translateY(0); }
                }
            `}</style>
        </div>
    );
};

export default Layout;
