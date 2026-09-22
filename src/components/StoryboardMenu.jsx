import React, { useId, useLayoutEffect, useRef, useState } from 'react';
import { BookOpen, ChevronDown } from 'lucide-react';
import { STORYBOARDS } from '../storyboards/index.js';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { useTrainingStore } from '../stores/trainingStore.js';
import NavButton from './NavButton.jsx';
import { startArchitectureSession } from '../stores/architectureHistory.js';

const StoryboardMenu = () => {
    const [open, setOpen] = useState(false);
    const rootRef = useRef(null);
    const menuRef = useRef(null);
    const menuId = useId();
    const architectureStore = useArchitectureStore;
    const templatesStore = useTemplatesStore;
    const trainingStore = useTrainingStore;

    useLayoutEffect(() => {
        if (!open) return;
        const trigger = rootRef.current.querySelector('button');
        const menu = menuRef.current;
        const position = () => {
            const anchor = trigger.getBoundingClientRect();
            const top = anchor.bottom + 6;
            menu.style.left = `${Math.max(8, Math.min(anchor.left, window.innerWidth - menu.offsetWidth - 8))}px`;
            menu.style.top = `${top}px`;
            menu.style.maxHeight = `${Math.max(0, window.innerHeight - top - 8)}px`;
        };
        const dismissOutside = (event) => {
            if (!rootRef.current.contains(event.target)) setOpen(false);
        };
        const dismissEscape = (event) => {
            if (event.key === 'Escape') {
                setOpen(false);
                trigger.focus();
            }
        };
        position();
        window.addEventListener('resize', position);
        document.addEventListener('pointerdown', dismissOutside);
        document.addEventListener('keydown', dismissEscape);
        return () => {
            window.removeEventListener('resize', position);
            document.removeEventListener('pointerdown', dismissOutside);
            document.removeEventListener('keydown', dismissEscape);
        };
    }, [open]);

    return (
        <div ref={rootRef} className="relative">
            <NavButton icon={BookOpen} label="Examples" onClick={() => setOpen((v) => !v)} title="Load a pre-built example" aria-expanded={open} aria-controls={menuId}>
                <ChevronDown size={11} strokeWidth={2.5} className="ml-0.5" />
            </NavButton>
            {open && (
                <div
                    ref={menuRef}
                    id={menuId}
                    role="group"
                    aria-label="Example architectures"
                    className="df-panel-subtle df-examples-menu"
                >
                    {STORYBOARDS.map((sb) => (
                        <button
                            key={sb.id}
                            onClick={() => {
                                architectureStore.getState().goToLevel(0);
                                startArchitectureSession(() => sb.run({ architectureStore, templatesStore, trainingStore }));
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
