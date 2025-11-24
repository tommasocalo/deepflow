import React, { useState, useRef } from 'react';
import {
    Box,
    Cpu,
    Activity,
    Layers,
    ArrowRightCircle,
    GitMerge,
    Maximize,
    Minimize,
    Repeat,
    Shuffle,
    Search,
    Plus,
    Wand2,
    StickyNote,
    Undo,
    Redo
} from 'lucide-react';

const RightSidebar = ({ onAddNode }) => {
    const [searchValue, setSearchValue] = useState('');
    const inputRef = useRef(null);

    const addNode = (nodeType) => {
        if (onAddNode) {
            // Add node at a default position (center-ish) with slight random offset
            const position = {
                x: 400 + Math.random() * 50,
                y: 300 + Math.random() * 50
            };
            onAddNode(nodeType, position);
        }
    };

    const handleSearchClick = () => {
        if (inputRef.current) {
            inputRef.current.focus();
        }
    };

    return (
        <div className="absolute pointer-events-none transition-all" style={{ right: '1rem', top: '1rem', zIndex: 40, width: '210px' }}>
            <div className="grid grid-cols-2 gap-2 pb-4 pointer-events-auto">
                {/* Search Bar - Actual Input Field */}
                <div
                    onClick={handleSearchClick}
                    className="col-span-2 flex items-center gap-2 px-2 py-1.5 border-2 border-black rounded-md shadow-medium bg-white cursor-text"
                >
                    <Search size={16} className="flex-shrink-0" />
                    <input
                        ref={inputRef}
                        type="text"
                        placeholder="Search nodes"
                        value={searchValue}
                        onChange={(e) => setSearchValue(e.target.value)}
                        className="flex-1 text-xs bg-transparent outline-none border-none ring-0 focus:ring-0 focus:outline-none appearance-none p-0 m-0 w-full h-full placeholder:text-gray-400"
                    />
                </div>

                {/* Add New - Pink */}
                <button
                    onClick={() => addNode('New')}
                    className="bg-pink-500 text-white border-2 border-black rounded-md font-bold shadow-medium hover:bg-pink-600 transition-colors py-2 px-2 text-xs flex items-center justify-center gap-1 h-[40px]"
                >
                    <Plus size={14} strokeWidth={2.5} />
                    Add New
                </button>

                {/* Input */}
                <button
                    onClick={() => addNode('Input')}
                    className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]"
                >
                    <Layers size={14} />
                    Input
                </button>

                {/* Dense */}
                <button onClick={() => addNode('Dense')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Box size={14} />
                    Dense
                </button>

                {/* Conv2D */}
                <button onClick={() => addNode('Conv2D')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Cpu size={14} />
                    Conv2D
                </button>

                {/* ReLU */}
                <button onClick={() => addNode('Activation')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Activity size={14} />
                    ReLU
                </button>

                {/* LSTM */}
                <button onClick={() => addNode('LSTM')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Repeat size={14} />
                    LSTM
                </button>

                {/* Dropout */}
                <button onClick={() => addNode('Dropout')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Minimize size={14} />
                    Dropout
                </button>

                {/* Batch */}
                <button onClick={() => addNode('BatchNorm')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Maximize size={14} />
                    Batch
                </button>

                {/* Pool */}
                <button onClick={() => addNode('Pool2D')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Shuffle size={14} />
                    Pool
                </button>

                {/* Concat */}
                <button onClick={() => addNode('Concat')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <GitMerge size={14} />
                    Concat
                </button>

                {/* Output */}
                <button onClick={() => addNode('Output')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <ArrowRightCircle size={14} />
                    Output
                </button>

                {/* Custom Prompt */}
                <button onClick={() => addNode('CustomPrompt')} className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Wand2 size={14} />
                    Custom
                </button>

                {/* Custom Node - Full Width */}
                <button onClick={() => addNode('CustomNode')} className="col-span-2 bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <StickyNote size={14} />
                    Custom Node
                </button>

                {/* Undo */}
                <button className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Undo size={14} />
                    Undo
                </button>

                {/* Redo */}
                <button className="bg-white border-2 border-black rounded-md shadow-medium hover:bg-cyan-300 transition-colors py-2 px-2 text-xs font-bold flex items-center justify-center gap-1 h-[40px]">
                    <Redo size={14} />
                    Redo
                </button>
            </div>
        </div>
    );
};

export default RightSidebar;
