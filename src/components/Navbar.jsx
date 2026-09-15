import { Workflow } from 'lucide-react';
import StoryboardMenu from './StoryboardMenu.jsx';

export default function Navbar() {
    return (
        <header className="df-navbar">
            <div className="df-brand">
                <span className="df-brand-icon"><Workflow size={19} strokeWidth={1.9} /></span>
                <span className="df-brand-name">DeepFlow</span>
                <span className="df-workspace-name">Model workspace</span>
            </div>
            <div className="df-navbar-actions">
                <span className="df-local-state" title="Your graph is stored in this browser. Use Save to download a copy."><span />Local workspace</span>
                <StoryboardMenu />
            </div>
        </header>
    );
}
