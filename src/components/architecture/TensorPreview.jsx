import { useId } from 'react';
import { activationLabel } from './nodeGeometry.js';
import './tensor-preview.css';

const MAX_ROWS = 6;
const MAX_COLUMNS = 24;

function groupedAxis(size, preferredGroupSize, limit) {
    if (!Number.isInteger(size) || size < 1) return { count: 1, groupSize: null };
    const groupSize = Math.max(preferredGroupSize, Math.ceil(size / limit));
    return { count: Math.min(limit, Math.ceil(size / groupSize)), groupSize };
}

/** A bounded shape sketch. Uniform tiles encode dimensions, never tensor values. */
export default function TensorPreview({ shape, compact = false, contrast = 'light' }) {
    const descriptionId = useId();
    const label = activationLabel(shape);
    if (!label) return null;
    const features = shape[shape.length - 1];
    const sequence = shape.length > 1 ? shape[shape.length - 2] : 1;
    const columns = groupedAxis(features, 128, MAX_COLUMNS);
    const rows = groupedAxis(sequence, 256, MAX_ROWS);
    const rowAxis = shape.length === 3 ? 'tokens' : 'entries on the preceding axis';
    const cell = compact ? 3 : 4;
    const step = cell + 1;
    const width = columns.count * step + 1;
    const height = rows.count * step + 1;
    const description = `${label}. ${rows.groupSize ? `Each row groups up to ${Math.min(rows.groupSize, sequence)} ${rowAxis}` : 'The preceding dimension is unknown'}; ${columns.groupSize ? `each column groups up to ${Math.min(columns.groupSize, features)} features` : 'the feature dimension is unknown'}. Uniform cells show shape only; activation values are not displayed.`;

    return (
        <span className={`df-tensor-preview${compact ? ' is-compact' : ''}${contrast === 'dark' ? ' is-dark' : ''}`} title={description}>
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={descriptionId} className="df-tensor-grid">
                <title id={descriptionId}>{description}</title>
                {Array.from({ length: rows.count }, (_, row) => Array.from({ length: columns.count }, (_, column) => (
                    <rect key={`${row}-${column}`} x={1 + column * step} y={1 + row * step} width={cell} height={cell} rx="0.4" />
                )))}
            </svg>
            <span className="df-tensor-dimensions" aria-hidden="true">{label}</span>
        </span>
    );
}
