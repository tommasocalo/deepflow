import React, { useState } from 'react';
import { Group, Trash2, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';

/**
 * Floats over the canvas when 2+ nodes are selected. Phase 3 actions:
 *   - Group selection into a new template + instance
 *   - Delete selection
 *   - Clear selection
 *
 * Phase 4 will add a "repeat with variation" action.
 */
const SelectionToolbar = () => {
    const selectedNodeIds = useArchitectureStore((s) => s.selectedNodeIds);
    const groupSelection = useArchitectureStore((s) => s.groupSelection);
    const removeNode = useArchitectureStore((s) => s.removeNode);
    const clearSelection = useArchitectureStore((s) => s.clearSelection);
    const enterSupernode = useArchitectureStore((s) => s.enterSupernode);
    const [namePrompt, setNamePrompt] = useState(null);

    if (selectedNodeIds.length < 2) return null;

    const handleGroup = () => setNamePrompt('Supernode');

    const confirmGroup = () => {
        const name = (namePrompt || 'Supernode').trim() || 'Supernode';
        const nodeIds = [...selectedNodeIds];
        const tplId = groupSelection(nodeIds, name);
        setNamePrompt(null);
        if (!tplId) {
            // groupSelection returns null when the selection doesn't have
            // a clean 1-in / 1-out boundary. Phase 4 will support this.
            return;
        }
    };

    const handleDelete = () => {
        for (const id of selectedNodeIds) removeNode(id);
        clearSelection();
    };

    return (
        <div
            className="df-panel-subtle absolute pointer-events-auto flex flex-col overflow-hidden"
            style={{ left: '50%', bottom: '1.5rem', transform: 'translateX(-50%)', zIndex: 50 }}
        >
            <div className="flex items-stretch">
                <span className="bg-white px-3 py-1.5 text-xs font-bold flex items-center gap-1 border-r border-black/15">
                    {selectedNodeIds.length} selected
                </span>
                <button
                    onClick={handleGroup}
                    className="bg-emerald-100 px-3 py-1.5 text-xs font-bold hover:bg-emerald-200 flex items-center gap-1 border-r border-black/15"
                    title="Group selection into a new supernode template"
                >
                    <Group size={12} strokeWidth={3} /> Group
                </button>
                <button
                    onClick={handleDelete}
                    className="bg-red-50 px-3 py-1.5 text-xs font-bold hover:bg-red-100 flex items-center gap-1 border-r border-black/15"
                    title="Delete selection"
                >
                    <Trash2 size={12} strokeWidth={3} /> Delete
                </button>
                <button
                    onClick={clearSelection}
                    className="bg-white px-3 py-1.5 text-xs font-bold hover:bg-stone-100"
                    title="Clear selection"
                >
                    <X size={12} strokeWidth={3} />
                </button>
            </div>

            {namePrompt !== null && (
                <div className="bg-white px-3 py-2 border-t border-black/15 flex items-center gap-2">
                    <label className="text-xs font-bold whitespace-nowrap">Name:</label>
                    <input
                        autoFocus
                        type="text"
                        value={namePrompt}
                        onChange={(e) => setNamePrompt(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter') confirmGroup();
                            if (e.key === 'Escape') setNamePrompt(null);
                        }}
                        className="df-input flex-1 px-2 py-1 text-xs"
                    />
                    <button
                        onClick={confirmGroup}
                        className="bg-black text-white px-2 py-1 text-xs font-bold rounded"
                    >
                        OK
                    </button>
                </div>
            )}
        </div>
    );
};

export default SelectionToolbar;
