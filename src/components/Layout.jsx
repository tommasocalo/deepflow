import { useState } from 'react';
import Navbar from './Navbar.jsx';
import ToolbarStrip from './ToolbarStrip.jsx';
import ArchitectureCanvas from './architecture/ArchitectureCanvas.jsx';
import BlockPalette from './architecture/BlockPalette.jsx';
import Breadcrumb from './architecture/Breadcrumb.jsx';
import SelectionToolbar from './architecture/SelectionToolbar.jsx';
import TrainingCanvas from './training/TrainingCanvas.jsx';
import TrainingPalette from './training/TrainingPalette.jsx';
import CodePanel from './CodePanel.jsx';

export default function Layout() {
    const [activeTab, setActiveTab] = useState('architecture');
    const [codeOpen, setCodeOpen] = useState(false);
    return (
        <div className="df-app">
            <Navbar />
            <ToolbarStrip activeTab={activeTab} onChange={setActiveTab} onShowCode={() => setCodeOpen(true)} />
            <main className="df-app-main" aria-label={activeTab === 'architecture' ? 'Architecture editor' : 'Training editor'}>
                {activeTab === 'architecture' && <ArchitectureCanvas overlay={<><Breadcrumb /><SelectionToolbar /></>}><BlockPalette /></ArchitectureCanvas>}
                {activeTab === 'training' && <><TrainingCanvas /><TrainingPalette /></>}
                <CodePanel open={codeOpen} onClose={() => setCodeOpen(false)} />
            </main>
        </div>
    );
}
