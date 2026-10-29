import { Brain } from 'lucide-react';
import StoryboardMenu from './StoryboardMenu.jsx';
import { PRODUCT_NAME } from '../edu/config.js';

export default function Navbar() {
    return (
        <header className="df-navbar">
            <div className="df-brand">
                <Brain size={22} strokeWidth={2.5} />
                <span className="df-brand-name">{PRODUCT_NAME}</span>
            </div>
            <div className="df-navbar-divider" />
            <StoryboardMenu />
        </header>
    );
}
