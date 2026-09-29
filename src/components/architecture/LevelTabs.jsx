import React, { useMemo } from 'react';
import { Boxes, Home, Repeat, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { resolveLevelContext } from '../../hooks/useValidation.js';
import { useOpenToSideMenu } from './OpenToSideContext.jsx';
import './level-tabs.css';

function LevelCard({ card, active, onOpen, onClose }) {
    const menuProps = useOpenToSideMenu(card.path);
    const Icon = card.kind === 'root' ? Home : card.kind === 'repeat' ? Repeat : Boxes;
    return (
        <div className={`df-level-card${active ? ' is-active' : ''}`} onContextMenu={menuProps.onContextMenu}>
            <button
                type="button"
                role="tab"
                aria-selected={active}
                tabIndex={active ? 0 : -1}
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
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const templates = useTemplatesStore((s) => s.templates);
    const openLevel = useArchitectureStore((s) => s.openLevel);
    const closeLevel = useArchitectureStore((s) => s.closeLevel);

    const activeKey = JSON.stringify(currentPath);
    const cards = useMemo(() => {
        const rootLabel = rootGraph.importSource?.displayName
            || rootGraph.importSource?.repoId?.split('/').at(-1)
            || 'main';
        const views = [{ key: '[]', path: [], label: rootLabel, kind: 'root', title: `Top level · ${rootLabel}` }];
        for (const path of openPaths ?? []) {
            const level = resolveGraphLevel(rootGraph, path, templates, inspectedIterations);
            if (level.isRoot) continue;
            const parent = resolveGraphLevel(rootGraph, path.slice(0, -1), templates, inspectedIterations);
            const node = parent.graph.nodes[path.at(-1)];
            const count = node.kind === 'repeat'
                ? resolveLevelContext(rootGraph, path, templates, inspectedIterations).repeat?.count || '?'
                : null;
            views.push({
                key: JSON.stringify(path), path, label: level.label, kind: node.kind, count,
                title: [rootLabel, ...level.segments.map((segment) => segment.label)].join(' / '),
            });
        }
        return views;
    }, [rootGraph, openPaths, templates, inspectedIterations]);

    return (
        <div className="df-levels" role="tablist" aria-label="Open levels" onKeyDown={(event) => {
            if (event.target.getAttribute('role') !== 'tab' || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            event.preventDefault();
            const tabs = [...event.currentTarget.querySelectorAll('[role="tab"]')];
            const index = tabs.indexOf(event.target);
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
            tabs[next]?.focus();
            tabs[next]?.click();
        }}>
            {cards.map((card, i) => (
                <LevelCard key={card.key} card={card} active={card.key === activeKey} onOpen={openLevel} onClose={i > 0 ? closeLevel : null} />
            ))}
        </div>
    );
}
