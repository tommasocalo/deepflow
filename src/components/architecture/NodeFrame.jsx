import { useId } from 'react';
import { Handle, Position } from '@xyflow/react';
import { AlertCircle, AlertTriangle, ChevronDown, ChevronUp, X } from 'lucide-react';
import { formatShape } from '../../validation/index.js';

export default function NodeFrame({
    label, category, icon, accent = '#4f46e5', tint = '#eef2ff',
    selected, expanded, onToggle, onRename, onRemove, onOpen,
    subtitle, summary, children, errors = [], warnings = [], outputShape,
    hasInput = true, hasOutput = true, repeated = false, badge,
}) {
    const contentId = useId();
    const Icon = icon;
    const status = errors.length ? 'error' : warnings.length ? 'warning' : '';
    const IssueIcon = errors.length ? AlertCircle : AlertTriangle;
    const issueCount = errors.length || warnings.length;
    const stopEvent = (event) => event.stopPropagation();

    return (
        <div
            className={`df-node ${selected ? 'is-selected' : ''} ${expanded ? 'is-expanded' : ''} ${repeated ? 'is-repeated' : ''} ${status ? `has-${status}` : ''}`}
            style={{ '--node-accent': accent, '--node-tint': tint }}
            onDoubleClick={onOpen ? (event) => {
                if (event.target.closest('button, input, select, textarea, label, summary')) return;
                event.stopPropagation();
                onOpen();
            } : undefined}
        >
            <button
                type="button"
                aria-label={`Delete ${label || 'block'}`}
                title="Delete block"
                className="nodrag nopan df-node-delete"
                onMouseDown={stopEvent}
                onClick={(event) => { event.stopPropagation(); onRemove(); }}
            >
                <X size={12} />
            </button>

            <div className="df-node-heading">
                <span className="df-node-icon"><Icon size={17} strokeWidth={1.8} /></span>
                <div className="df-node-heading-text">
                    <div className="df-node-eyebrow">
                        <span>{category}</span>
                        {badge && <span className="df-node-count">{badge}</span>}
                    </div>
                    {expanded ? (
                        <input
                            aria-label="Block name"
                            value={label}
                            onChange={(event) => onRename(event.target.value)}
                            onMouseDown={stopEvent}
                            onKeyDown={stopEvent}
                            className="nodrag nopan df-node-name-input"
                            spellCheck={false}
                        />
                    ) : (
                        <h3 className="df-node-title" title={label}>{label || 'Untitled block'}</h3>
                    )}
                </div>
                <button
                    type="button"
                    aria-label={`${expanded ? 'Collapse' : 'Expand'} ${label || 'block'}`}
                    aria-expanded={expanded}
                    aria-controls={contentId}
                    title={expanded ? 'Collapse parameters' : 'Edit parameters'}
                    className="nodrag nopan df-node-expand"
                    onMouseDown={stopEvent}
                    onClick={(event) => { event.stopPropagation(); onToggle(); }}
                >
                    {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
            </div>

            {subtitle && <div className="df-node-subtitle">{subtitle}</div>}
            {!expanded && summary}
            <div id={contentId} hidden={!expanded}>
                {expanded && (
                    <div
                        className="nodrag nopan nowheel df-node-body"
                        onMouseDown={stopEvent}
                        onClick={stopEvent}
                        onDoubleClick={stopEvent}
                        onKeyDown={stopEvent}
                    >
                        {children}
                        {errors.length > 0 && <Issues messages={errors} kind="error" />}
                        {warnings.length > 0 && <Issues messages={warnings} kind="warning" />}
                    </div>
                )}
            </div>

            {(outputShape || status) && (
                <div className="df-node-footer">
                    {outputShape && (
                        <div className="df-node-shape" title={`Output: ${formatShape(outputShape)}`}>
                            <span>OUT</span><code>{formatShape(outputShape)}</code>
                        </div>
                    )}
                    {status && (
                        <button
                            type="button"
                            className={`nodrag nopan df-node-status is-${status}`}
                            title={[...errors, ...warnings].join('\n')}
                            aria-label={`${issueCount} ${status}${issueCount === 1 ? '' : 's'}: show details`}
                            onMouseDown={stopEvent}
                            onClick={(event) => { event.stopPropagation(); if (!expanded) onToggle(); }}
                        >
                            <IssueIcon size={12} /><span>{issueCount}</span>
                        </button>
                    )}
                </div>
            )}
            {hasInput && <Handle type="target" position={Position.Top} className="df-node-handle" title="Input connection" />}
            {hasOutput && <Handle type="source" position={Position.Bottom} className="df-node-handle" title="Output connection" />}
        </div>
    );
}

function Issues({ messages, kind }) {
    const Icon = kind === 'error' ? AlertCircle : AlertTriangle;
    return (
        <div className={`df-node-issues is-${kind}`}>
            <div><Icon size={13} /><strong>{kind === 'error' ? 'Needs attention' : 'Warnings'}</strong></div>
            <ul>{messages.map((message, index) => <li key={index}>{message}</li>)}</ul>
        </div>
    );
}
