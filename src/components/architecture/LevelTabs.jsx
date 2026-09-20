import React from 'react';
import { Boxes, Home, Repeat, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useOpenToSideMenu } from './OpenToSideContext.jsx';

function LevelCard({ card, active, onOpen, onClose }) {
    const menuProps = useOpenToSideMenu(card.path);
    const Icon = card.kind === 'root' ? Home : card.kind === 'repeat' ? Repeat : Boxes;
    return (
        <div className={`df-level-card${active ? ' is-active' : ''}`} onContextMenu={menuProps.onContextMenu}>
            <button
                type="button"
                role="tab"
                aria-selected={active}
                aria-keyshortcuts={menuProps['aria-keyshortcuts']}
                className="df-level-tab"
                onClick={() => onOpen(card.path)}
                onKeyDown={menuProps.onKeyDown}
                title={card.title}
            >
                <Icon size={12} strokeWidth={2.5} />
                <span className="df-level-label">{card.label}</span>
                {card.count != null && <span className="df-level-count">× {card.count}</span>}
            </button>
            {onClose && (
                <button
                    type="button"
                    className="df-level-close"
                    onClick={() => onClose(card.path)}
                    aria-label={`Close ${card.label}`}
                    title="Close"
                >
                    <X size={11} strokeWidth={3} />
                </button>
            )}
        </div>
    );
}

/**
 * Open graph views stay in the strip independently of the active view.
 * Selecting main or another supernode does not close any cards.
 */
export default function LevelTabs() {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const openPaths = useArchitectureStore((s) => s.openPaths);
    const templates = useTemplatesStore((s) => s.templates);
    const openLevel = useArchitectureStore((s) => s.openLevel);
    const closeLevel = useArchitectureStore((s) => s.closeLevel);

    const activeKey = JSON.stringify(currentPath);
    const cards = [{ key: '[]', path: [], label: 'main', kind: 'root', title: 'Top level' }];
    for (const path of openPaths ?? []) {
        const level = resolveGraphLevel(rootGraph, path, templates);
        if (level.isRoot) continue;
        const parent = resolveGraphLevel(rootGraph, path.slice(0, -1), templates);
        const node = parent.graph.nodes[path.at(-1)];
        cards.push({
            key: JSON.stringify(path),
            path,
            label: level.label,
            kind: node.kind,
            count: node?.kind === 'repeat' ? node.count : null,
            title: ['main', ...level.segments.map((segment) => segment.label)].join(' / '),
        });
    }

    return (
        <div className="df-levels" role="tablist" aria-label="Open levels">
            {cards.map((card, i) => (
                <LevelCard key={card.key} card={card} active={card.key === activeKey} onOpen={openLevel} onClose={i > 0 ? closeLevel : null} />
            ))}
            {currentPath.length > 0 && (
                <span className="df-levels-hint">Edits inside a template apply to every instance.</span>
            )}
        </div>
    );
}
