import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { Boxes, ChevronDown, Columns2, Home, Repeat, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { resolveLevelContext } from '../../hooks/useValidation.js';
import { useOpenToSide, useOpenToSideMenu } from './OpenToSideContext.jsx';
import './level-tabs.css';

function LevelRow({ card, active, onOpen, onClose, onOpenToSide }) {
    const sideMenu = useOpenToSideMenu(card.path);
    const Icon = card.kind === 'root' ? Home : card.kind === 'repeat' ? Repeat : Boxes;
    return <div className={`df-level-row${active ? ' is-active' : ''}`}>
        <button type="button" className="df-level-entry" aria-current={active ? 'page' : undefined}
            aria-keyshortcuts={sideMenu['aria-keyshortcuts']} onClick={() => onOpen(card.path)}
            onContextMenu={sideMenu.onContextMenu} onKeyDown={sideMenu.onKeyDown}
            title={card.title}>
            <Icon size={14} strokeWidth={2.2} aria-hidden="true" />
            <span className="df-level-entry-text"><strong>{card.label}</strong><small>{card.title}</small></span>
            {card.count != null && <span className="df-level-count">× {card.count}</span>}
        </button>
        {onOpenToSide && <button type="button" className="df-level-row-action" onClick={() => onOpenToSide(card.path)}
            aria-label={`Open ${card.label} to the side`} title="Open to the side"><Columns2 size={14} aria-hidden="true" /></button>}
        {onClose && <button type="button" className="df-level-row-action df-level-row-close" onClick={() => onClose(card.path)}
            aria-label={`Close ${card.label}`} title="Close level"><X size={14} aria-hidden="true" /></button>}
    </div>;
}

/** A compact per-pane switcher; open graph levels remain independent of the active view. */
export default function LevelTabs() {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const openPaths = useArchitectureStore((s) => s.openPaths);
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const templates = useTemplatesStore((s) => s.templates);
    const openLevel = useArchitectureStore((s) => s.openLevel);
    const closeLevel = useArchitectureStore((s) => s.closeLevel);
    const openToSide = useOpenToSide();
    const sideMenu = useOpenToSideMenu(currentPath);
    const [expanded, setExpanded] = useState(false);
    const controlRef = useRef(null);
    const triggerRef = useRef(null);
    const panelId = useId();

    const activeKey = JSON.stringify(currentPath);
    const cards = useMemo(() => {
        const rootLabel = rootGraph.importSource?.displayName
            || rootGraph.importSource?.repoId?.split('/').at(-1)
            || 'main';
        const views = [{ key: '[]', path: [], label: rootLabel, kind: 'root', title: rootLabel }];
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
    const current = cards.find((card) => card.key === activeKey) ?? cards[0];
    const currentLevel = resolveGraphLevel(rootGraph, currentPath, templates, inspectedIterations);
    const crumbs = [cards[0], ...currentLevel.segments.map((segment, index) => ({
        key: JSON.stringify(currentPath.slice(0, index + 1)),
        path: currentPath.slice(0, index + 1),
        label: segment.label,
    }))];

    useEffect(() => {
        if (!expanded) return undefined;
        const dismiss = (event) => { if (!controlRef.current?.contains(event.target)) setExpanded(false); };
        document.addEventListener('pointerdown', dismiss, true);
        return () => document.removeEventListener('pointerdown', dismiss, true);
    }, [expanded]);

    const show = (focusFirst = false) => {
        setExpanded(true);
        if (focusFirst) requestAnimationFrame(() => controlRef.current?.querySelector('.df-level-entry')?.focus({ preventScroll: true }));
    };
    const hide = (restoreFocus = false) => {
        setExpanded(false);
        if (restoreFocus) requestAnimationFrame(() => triggerRef.current?.focus({ preventScroll: true }));
    };
    const navigate = (path) => { openLevel(path); hide(true); };
    const closeView = (path) => {
        const wasCurrent = JSON.stringify(path) === activeKey;
        closeLevel(path);
        if (wasCurrent) hide(true);
    };
    const openSide = (path) => { openToSide?.(path); hide(); };

    return <div ref={controlRef} className="df-level-control nodrag nopan nowheel" data-level-control
        onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setExpanded(false); }}
        onKeyDown={(event) => {
            if (event.key === 'Escape' && expanded) { event.preventDefault(); event.stopPropagation(); hide(true); }
        }}>
        <button ref={triggerRef} type="button" className="df-level-trigger" data-level-trigger
            aria-label={`Current level: ${current.title}. Show open levels`}
            aria-haspopup="dialog" aria-expanded={expanded} aria-controls={expanded ? panelId : undefined}
            title={current.title} onClick={() => setExpanded((value) => !value)}
            onContextMenu={sideMenu.onContextMenu} onKeyDown={(event) => {
                sideMenu.onKeyDown(event);
                if (!event.defaultPrevented && event.key === 'ArrowDown') { event.preventDefault(); show(true); }
            }}>
            {current.kind === 'root' ? <Home size={13} strokeWidth={2.4} aria-hidden="true" />
                : current.kind === 'repeat' ? <Repeat size={13} strokeWidth={2.4} aria-hidden="true" />
                    : <Boxes size={13} strokeWidth={2.4} aria-hidden="true" />}
            <span className="df-level-trigger-name">{current.label}</span>
            {current.count != null && <span className="df-level-count">× {current.count}</span>}
            <ChevronDown size={12} strokeWidth={2.4} className={expanded ? 'is-open' : ''} aria-hidden="true" />
        </button>
        {expanded && <div id={panelId} className="df-level-panel" role="dialog" aria-label="Open levels">
            <div className="df-level-panel-header"><span>Current level</span><button type="button" aria-label="Close levels menu" onClick={() => hide(true)}><X size={13} aria-hidden="true" /></button></div>
            <div className="df-level-breadcrumb" aria-label="Current level path">
                {crumbs.map((crumb, index) => <span key={crumb.key} className="df-level-crumb">
                    {index > 0 && <span aria-hidden="true">/</span>}
                    <button type="button" aria-current={index === crumbs.length - 1 ? 'page' : undefined}
                        onClick={() => navigate(crumb.path)}>{crumb.label}</button>
                </span>)}
            </div>
            <div className="df-level-panel-list" aria-label="Open views">
                <div className="df-level-panel-subtitle">Open views <span>{cards.length}</span></div>
                {cards.map((card, index) => <LevelRow key={card.key} card={card} active={card.key === activeKey}
                    onOpen={navigate} onClose={index ? closeView : null} onOpenToSide={openToSide ? openSide : null} />)}
            </div>
        </div>}
    </div>;
}
