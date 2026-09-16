import { useId } from 'react';
import { activationLabel } from './nodeGeometry.js';
import { featurePreview } from './featurePreview.js';
import './tensor-preview.css';

/** Feature-vector slice, grouped without inventing numerical activation values. */
export default function TensorPreview({ shape, featureAxis = shape?.length - 1, featureLabel = 'features' }) {
    const descriptionId = useId();
    const preview = featurePreview(shape, featureAxis);
    if (!preview) return null;
    const { features, groupSize, count, columns, width, height } = preview;
    const tokenSlice = shape.length === 3 && featureAxis === 2;
    const context = tokenSlice ? `${shape[1] ?? '?'} tokens` : activationLabel(shape);
    const description = `${activationLabel(shape)}. One ${tokenSlice ? 'token' : 'position'} across ${features} ${featureLabel}. Each square groups up to ${groupSize} adjacent ${featureLabel}; the last square may contain fewer. Shape preview; numerical activation values are not loaded.`;
    return (
        <div className="df-tensor-preview" title={description}>
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={descriptionId} className="df-tensor-grid">
                <title id={descriptionId}>{description}</title>
                {Array.from({ length: count }, (_, index) => (
                    <rect key={index} x={(index % columns) * 7} y={Math.floor(index / columns) * 7} width={6} height={6} rx=".4" />
                ))}
            </svg>
            <div className="df-tensor-key" aria-hidden="true">
                <strong className="df-tensor-features">{features.toLocaleString('en-US')}</strong>
                <span className="df-tensor-grouping"><i /> ×{groupSize.toLocaleString('en-US')}</span>
            </div>
            <span className="sr-only">{context} · {features} {featureLabel}</span>
        </div>
    );
}
