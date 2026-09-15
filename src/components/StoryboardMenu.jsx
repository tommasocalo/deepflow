import React, { useState } from 'react';
import { BookOpen, ChevronDown } from 'lucide-react';
import { STORYBOARDS } from '../storyboards/index.js';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { useTrainingStore } from '../stores/trainingStore.js';
import NavButton from './NavButton.jsx';

const StoryboardMenu = () => {
    const [open, setOpen] = useState(false);
    const architectureStore = useArchitectureStore;
    const templatesStore = useTemplatesStore;
    const trainingStore = useTrainingStore;

    return (
        <div className="relative">
            <NavButton icon={BookOpen} label="Examples" onClick={() => setOpen((v) => !v)} title="Load a pre-built example">
                <ChevronDown size={11} strokeWidth={2.5} className="ml-0.5" />
            </NavButton>
            {open && (
                <div
                    className="df-panel-subtle absolute right-0 mt-1 w-72 z-50 overflow-hidden"
                    onMouseLeave={() => setOpen(false)}
                >
                    {STORYBOARDS.map((sb) => (
                        <button
                            key={sb.id}
                            onClick={() => {
                                sb.run({ architectureStore, templatesStore, trainingStore });
                                setOpen(false);
                            }}
                            className="w-full text-left p-2 border-b border-black/10 last:border-b-0 hover:bg-stone-50"
                        >
                            <div className="font-bold text-xs">{sb.label}</div>
                            <div className="text-[11px] text-text-secondary mt-0.5">{sb.summary}</div>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
};

export default StoryboardMenu;
