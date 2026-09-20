import { useState, useRef, useEffect, useId } from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, LocateFixed } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { groupGraphIssues } from './graphIssues.js';
import './graph-status.css';

function IssueMessages({ errors, warnings }) {
    return (
        <span className="df-graph-issue-messages">
            {['error', 'warning'].flatMap((tone) => (tone === 'error' ? errors : warnings).map((message, index) => (
                <span key={`${tone}:${index}`} className={`df-graph-issue-message is-${tone}`}>
                    {tone === 'error' ? <AlertCircle size={10} aria-hidden="true" /> : <AlertTriangle size={10} aria-hidden="true" />}
                    <span><span className="sr-only">{tone === 'error' ? 'Error: ' : 'Warning: '}</span>{message}</span>
                </span>
            )))}
        </span>
    );
}

/** Current-level checks, grouped by node; clicking a group opens its inspector. */
export default function GraphStatusPanel() {
    const validation = useValidation();
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const setExpanded = useArchitectureStore((s) => s.setExpanded);
    const archGraph = useArchitectureStore((s) => s.graph);
    const templates = useTemplatesStore((s) => s.templates);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const [open, setOpen] = useState(false);
    const [hoverNodeId, setHoverNodeId] = useState(null);
    const ref = useRef(null);
    const triggerRef = useRef(null);
    const issuesId = useId();
    const { setCenter, getZoom } = useReactFlow();
    const viewGraph = resolveGraphLevel(archGraph, currentPath, templates).graph;
    const groups = groupGraphIssues(validation, viewGraph, templates);
    const errorCount = groups.reduce((count, group) => count + group.errors.length, 0);
    const warningCount = groups.reduce((count, group) => count + group.warnings.length, 0);
    const hasIssues = groups.length > 0;
    const visible = open && hasIssues;

    useEffect(() => {
        if (!visible) return;
        const onDocumentMouseDown = (event) => {
            if (!ref.current?.contains(event.target)) setOpen(false);
        };
        document.addEventListener('mousedown', onDocumentMouseDown);
        return () => document.removeEventListener('mousedown', onDocumentMouseDown);
    }, [visible]);

    useEffect(() => {
        if (!visible || !hoverNodeId) return;
        // IDs may come from imported graphs; do not interpolate them into a selector.
        const canvas = ref.current?.closest('[data-architecture-canvas]');
        const element = Array.from(canvas?.querySelectorAll('.react-flow__node') ?? [])
            .find((candidate) => candidate.dataset.id === hoverNodeId);
        if (!element) return;
        element.classList.add('deepflow-hover-issue');
        return () => element.classList.remove('deepflow-hover-issue');
    }, [hoverNodeId, visible, currentPath]);

    const tone = errorCount ? 'error' : warningCount ? 'warning' : 'ok';
    const Icon = errorCount ? AlertCircle : warningCount ? AlertTriangle : CheckCircle2;
    const label = [
        errorCount && `${errorCount} error${errorCount === 1 ? '' : 's'}`,
        warningCount && `${warningCount} warning${warningCount === 1 ? '' : 's'}`,
    ].filter(Boolean).join(' · ') || 'No issues detected';

    const openNode = (nodeId) => {
        const node = viewGraph?.nodes?.[nodeId];
        if (!node) return;
        selectOnly(nodeId);
        setExpanded(nodeId);
        setCenter(node.position.x, node.position.y + 24, { duration: 300, zoom: Math.max(getZoom(), 1) });
        setHoverNodeId(null);
        setOpen(false);
    };

    return (
        <div ref={ref} className="df-graph-status" onKeyDown={(event) => {
            if (event.key === 'Escape' && visible) {
                event.stopPropagation();
                setOpen(false);
                setHoverNodeId(null);
                triggerRef.current?.focus();
            }
        }}>
            <button
                ref={triggerRef}
                type="button"
                onClick={() => setOpen((value) => !value)}
                disabled={!hasIssues}
                aria-expanded={visible}
                aria-controls={visible ? issuesId : undefined}
                aria-label={`Graph checks: ${label}`}
                className={`df-graph-status-trigger is-${tone}`}
            >
                <Icon size={12} strokeWidth={2.8} aria-hidden="true" />
                <span>{label}</span>
                {hasIssues && <ChevronDown size={11} strokeWidth={2.5} className={visible ? 'is-open' : ''} aria-hidden="true" />}
            </button>
            {visible && (
                <div id={issuesId} className="df-graph-status-popover" role="region" aria-label="Graph issues">
                    {groups.map((group) => group.nodeId ? (
                        <button
                            key={group.key}
                            type="button"
                            className="df-graph-issue-group"
                            data-issue-node-id={group.nodeId}
                            disabled={!viewGraph?.nodes?.[group.nodeId]}
                            onMouseEnter={() => setHoverNodeId(group.nodeId)}
                            onMouseLeave={() => setHoverNodeId(null)}
                            onFocus={() => setHoverNodeId(group.nodeId)}
                            onBlur={() => setHoverNodeId(null)}
                            onClick={() => openNode(group.nodeId)}
                        >
                            <span className="df-graph-issue-heading">
                                <span>{group.label}</span>
                                <LocateFixed size={11} aria-hidden="true" />
                            </span>
                            <IssueMessages errors={group.errors} warnings={group.warnings} />
                        </button>
                    ) : (
                        <div key={group.key} className="df-graph-issue-group df-graph-issue-group--global">
                            <span className="df-graph-issue-heading">Graph</span>
                            <IssueMessages errors={group.errors} warnings={group.warnings} />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
