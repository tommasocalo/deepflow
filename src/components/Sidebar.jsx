import React, { useState } from 'react';
import {
    LayoutGrid,
    Settings,
    Save,
    Download,
    Upload,
    Share2
} from 'lucide-react';

const SidebarItem = ({ icon: Icon, label, active, isExpanded, hasBorder = true }) => {
    return (
        <div
            className={`flex items-center py-1.5 px-2 cursor-pointer hover:bg-slate-100 transition-all whitespace-nowrap ${active ? 'bg-slate-100' : ''} ${hasBorder ? 'border-b border-slate-300' : ''}`}
        >
            <div className="min-w-[24px]">
                <Icon size={16} strokeWidth={2.5} />
            </div>
            {isExpanded && <span className="text-sm font-medium ml-2">{label}</span>}
        </div>
    );
};

const Sidebar = () => {
    const [isHovered, setIsHovered] = useState(false);

    return (
        <div
            className="absolute flex flex-col pointer-events-none transition-all"
            style={{ left: '1rem', top: '1rem', zIndex: 40 }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {/* Expandable Sidebar Panel */}
            <div
                className="bg-surface border-2 border-black rounded-md shadow-hard pointer-events-auto flex flex-col transition-all duration-200"
                style={{ width: isHovered ? '160px' : '36px', overflow: 'hidden' }}
            >
                <SidebarItem icon={LayoutGrid} label="Overview" active isExpanded={isHovered} />
                <SidebarItem icon={Settings} label="Settings" isExpanded={isHovered} />
                <SidebarItem icon={Save} label="Save" isExpanded={isHovered} />
                <SidebarItem icon={Download} label="Save As" isExpanded={isHovered} />
                <SidebarItem icon={Upload} label="Import" isExpanded={isHovered} />
                <SidebarItem icon={Share2} label="Export" isExpanded={isHovered} hasBorder={false} />
            </div>
        </div>
    );
};

export default Sidebar;
