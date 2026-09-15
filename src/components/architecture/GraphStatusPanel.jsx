import React, { useState, useRef, useEffect } from 'react';
import {
    AlertCircle,
    AlertTriangle,
    CheckCircle2,
    ChevronDown,
    LocateFixed,
} from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';

/**
 * Top-left chip showing the validation status. Click to expand the issue
 * list. Hovering an issue softly highlights the offending node by setting
 * it as the singular hover-id; clicking it selects, expands, and centres
 * the viewport on it.
 *
 * Mounted INSIDE the ReactFlowProvider so we can use `useReactFlow` for
 * `setCenter`.
 */
const GraphStatusPanel = () => {
    const validation = useValidation();
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const setExpanded = useArchitectureStore((s) => s.setExpanded);
    const archGraph = useArchitectureStore((s) => s.graph);
    const templates = useTemplatesStore((s) => s.templates);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const [open, setOpen] = useState(false);
    const [hoverNodeId, setHoverNodeId] = useState(null);
    const ref = useRef(null);
    const { setCenter, getZoom } = useReactFlow();

    // Resolve view graph for node lookup
    const viewGraph = (() => {
        if (!currentPath?.length) return archGraph;
        let g = archGraph;
        for (const sid of currentPath) {
            const node = g?.nodes?.[sid];
            if (!node || node.kind !== 'supernode') return archGraph;
            const tpl = templates?.[node.templateId];
            if (!tpl) return archGraph;
            g = tpl.graph;
        }
        return g;
    })();

    useEffect(() => {
        if (!open) return;
        const onDoc = (e) => {
            if (!ref.current?.contains(e.target)) setOpen(false);
        };
        document.addEventListener('mousedown', onDoc);
        return () => document.removeEventListener('mousedown', onDoc);
    }, [open]);

    // Apply hover-highlight by adding a CSS class to the node element.
    useEffect(() => {
        if (!hoverNodeId) return;
        const el = document.querySelector(`.react-flow__node[data-id="${hoverNodeId}"]`);
        if (!el) return;
        el.classList.add('deepflow-hover-issue');
        return () => el.classList.remove('deepflow-hover-issue');
    }, [hoverNodeId]);

    const errorCount = Object.values(validation.errors).reduce((a, e) => a + e.length, 0)
        + validation.globalErrors.length;
    const warningCount = Object.values(validation.warnings).reduce((a, w) => a + w.length, 0);
    const issues = [
        ...validation.globalErrors.map((m) => ({ tone: 'error', nodeId: null, msg: m })),
        ...Object.entries(validation.errors).flatMap(([nodeId, msgs]) =>
            msgs.map((m) => ({ tone: 'error', nodeId, msg: m })),
        ),
        ...Object.entries(validation.warnings).flatMap(([nodeId, msgs]) =>
            msgs.map((m) => ({ tone: 'warning', nodeId, msg: m })),
        ),
    ];

    const tone = errorCount > 0
        ? { ringColor: '#dc2626', bg: 'bg-red-50',     text: 'text-red-700',     icon: <AlertCircle size={13} className="text-red-700" strokeWidth={3} />,    label: `${errorCount} error${errorCount !== 1 ? 's' : ''}${warningCount ? ` · ${warningCount} warning${warningCount !== 1 ? 's' : ''}` : ''}` }
        : warningCount > 0
            ? { ringColor: '#f59e0b', bg: 'bg-amber-50',   text: 'text-amber-700',   icon: <AlertTriangle size={13} className="text-amber-700" strokeWidth={3} />, label: `${warningCount} warning${warningCount !== 1 ? 's' : ''}` }
            : { ringColor: '#10b981', bg: 'bg-emerald-50', text: 'text-emerald-700', icon: <CheckCircle2 size={13} className="text-emerald-700" strokeWidth={3} />, label: 'Graph valid' };

    const focusNode = (nodeId) => {
        const node = viewGraph?.nodes?.[nodeId];
        if (!node) return;
        const x = node.position.x + 120;  // ~half of node width
        const y = node.position.y + 60;
        setCenter(x, y, { duration: 450, zoom: Math.max(getZoom(), 1.18) });
    };

    return (
        <div
            ref={ref}
            className="absolute pointer-events-auto"
            style={{ left: '1rem', top: '1rem', zIndex: 40 }}
        >
            <button
                onClick={() => issues.length > 0 && setOpen((v) => !v)}
                disabled={issues.length === 0}
                className={`${tone.bg} ${tone.text} rounded-lg shadow-sm inline-flex items-center gap-2 text-xs font-bold tracking-wide ${
                    issues.length > 0 ? 'cursor-pointer hover:brightness-95' : 'cursor-default'
                }`}
                style={{
                    padding: '7px 12px',
                    border: `2px solid ${tone.ringColor}`,
                }}
            >
                {tone.icon}
                <span>{tone.label}</span>
                {issues.length > 0 && (
                    <ChevronDown
                        size={12}
                        strokeWidth={3}
                        className={`transition-transform ${open ? 'rotate-180' : ''}`}
                    />
                )}
            </button>
            {open && issues.length > 0 && (
                <div
                    className="df-panel-subtle mt-2 max-h-80 overflow-y-auto"
                    style={{
                        width: 220,
                        padding: 10,
                    }}
                >
                    {issues.map((issue, i) => (
                        <button
                            key={i}
                            onMouseEnter={() => setHoverNodeId(issue.nodeId)}
                            onMouseLeave={() => setHoverNodeId(null)}
                            onClick={() => {
                                if (issue.nodeId) {
                                    selectOnly(issue.nodeId);
                                    setExpanded(issue.nodeId);
                                    focusNode(issue.nodeId);
                                }
                                setOpen(false);
                            }}
                            className={[
                                'block w-full text-left rounded-md hover:bg-stone-50 transition-colors',
                                issue.tone === 'error'
                                    ? 'bg-red-50/70 border border-red-200 border-l-4 border-l-red-500'
                                    : 'bg-amber-50/80 border border-amber-200 border-l-4 border-l-amber-500',
                            ].join(' ')}
                            style={{ padding: '8px 10px', marginBottom: 6 }}
                        >
                            <div className="flex items-start gap-1.5">
                                <span
                                    className={[
                                        'mt-1 h-1.5 w-1.5 rounded-full shrink-0',
                                        issue.tone === 'error' ? 'bg-red-600' : 'bg-amber-500',
                                    ].join(' ')}
                                />
                                <div className="min-w-0 flex-1 text-[11px] leading-snug whitespace-normal break-words">
                                    {issue.msg}
                                </div>
                                {issue.nodeId && (
                                    <LocateFixed size={10} strokeWidth={2.6} className="mt-0.5 opacity-45 shrink-0" />
                                )}
                            </div>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
};

export default GraphStatusPanel;
