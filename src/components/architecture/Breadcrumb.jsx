import React from 'react';
import { ChevronRight, Home } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';

/**
 * Walks the currentPath and renders a clickable trail
 *   main › ResNet stage › Conv block
 * Clicking any segment jumps back to that level.
 */
const Breadcrumb = () => {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const templates = useTemplatesStore((s) => s.templates);
    const goToLevel = useArchitectureStore((s) => s.goToLevel);

    if (!currentPath.length) return null;

    /** @type {Array<{id: string, label: string, depth: number}>} */
    const segments = [{ id: 'root', label: 'main', depth: 0 }];
    segments.push(...resolveGraphLevel(rootGraph, currentPath, templates).segments);

    return (
        <div
            className="df-panel-subtle df-breadcrumb absolute pointer-events-auto flex items-center gap-1 px-2 py-1"
            style={{ left: '50%', top: '1rem', transform: 'translateX(-50%)', zIndex: 40 }}
        >
            {segments.map((seg, i) => (
                <React.Fragment key={`${seg.depth}:${seg.id}`}>
                    {i > 0 && <ChevronRight size={12} className="opacity-50" />}
                    <button
                        onClick={() => goToLevel(seg.depth)}
                        aria-current={i === segments.length - 1 ? 'location' : undefined}
                        className={`text-xs font-bold px-2 py-0.5 rounded ${
                            i === segments.length - 1
                                ? 'bg-black text-white'
                                : 'hover:bg-stone-200'
                        }`}
                    >
                        {i === 0 ? (
                            <span className="flex items-center gap-1"><Home size={10} /> {seg.label}</span>
                        ) : seg.label}
                    </button>
                </React.Fragment>
            ))}
        </div>
    );
};

export default Breadcrumb;
