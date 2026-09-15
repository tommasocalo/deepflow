import React from 'react';
import { Brain, LogIn, UserPlus } from 'lucide-react';
import StoryboardMenu from './StoryboardMenu.jsx';
import NavButton from './NavButton.jsx';

const Navbar = () => (
    <div
        className="w-full bg-surface border-b border-black/15 flex items-center justify-between z-30 relative"
        style={{ padding: '0.5rem 1rem', minHeight: 48 }}
    >
        <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
                <Brain size={22} strokeWidth={2.5} />
                <span className="text-base font-bold tracking-tight">DeepFlow</span>
            </div>
            <div className="w-px h-5 bg-black/15" />
            <StoryboardMenu />
        </div>

        <div className="flex items-center gap-2">
            <NavButton icon={LogIn}    label="Login" />
            <NavButton icon={UserPlus} label="Sign Up" tone="accent" />
        </div>
    </div>
);

export default Navbar;
