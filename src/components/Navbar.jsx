import React from 'react';
import { Brain } from 'lucide-react';

const Navbar = () => {
    return (
        <div className="w-full bg-surface border-b-2 border-black flex items-center justify-between z-30 relative" style={{ padding: '1rem 1.5rem' }}>
            {/* Logo */}
            <div className="flex items-center gap-sm">
                <Brain size={28} strokeWidth={2.5} />
                <span className="text-xl font-bold tracking-tight">DeepFlow</span>
            </div>

            {/* Right Side Actions */}
            <div className="flex items-center gap-sm">
                <button className="btn text-xs px-3 py-1">Login</button>
                <button className="btn-accent text-xs px-3 py-1">Sign Up</button>
            </div>
        </div>
    );
};

export default Navbar;
