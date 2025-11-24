import React from 'react';
import { MoreHorizontal, X } from 'lucide-react';

const Block = ({ id, type, x, y, onDelete }) => {
    return (
        <div
            className="absolute bg-surface border-2 border-black rounded-md shadow-hard w-48 flex flex-col group transition-transform active:scale-95"
            style={{ left: x, top: y }}
        >
            {/* Header */}
            <div className="p-sm border-b-2 border-black flex items-center justify-between bg-surface">
                <span className="text-sm font-bold uppercase tracking-wider">{type}</span>
                <div className="flex items-center gap-1">
                    <X size={16} className="cursor-pointer hover:text-accent" onClick={() => onDelete(id)} strokeWidth={2.5} />
                </div>
            </div>

            {/* Body */}
            <div className="p-md flex flex-col gap-2">
                <div className="flex justify-between items-center">
                    <span className="text-xs font-bold text-text-secondary">Neurons</span>
                    <span className="text-xs font-bold border-2 border-black px-1 rounded bg-surface-hover">64</span>
                </div>
                <div className="flex justify-between items-center">
                    <span className="text-xs font-bold text-text-secondary">Activation</span>
                    <span className="text-xs font-bold border-2 border-black px-1 rounded bg-surface-hover">ReLU</span>
                </div>
            </div>

            {/* Ports */}
            <div className="absolute -left-1.5 top-1/2 w-3 h-3 bg-white border-2 border-black rounded-full transform -translate-y-1/2 hover:bg-black transition-colors cursor-pointer"></div>
            <div className="absolute -right-1.5 top-1/2 w-3 h-3 bg-white border-2 border-black rounded-full transform -translate-y-1/2 hover:bg-black transition-colors cursor-pointer"></div>
        </div>
    );
};

export default Block;
