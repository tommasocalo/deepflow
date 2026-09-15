/** Category colors shared by the library and architecture cards. */
const STYLES = {
    IO:               { accent: '#64748b', tint: '#f1f5f9' },
    Convolution:      { accent: '#059669', tint: '#ecfdf5' },
    Pooling:          { accent: '#3f8b65', tint: '#f0fdf4' },
    Linear:           { accent: '#3b6bc5', tint: '#eff6ff' },
    Reshape:          { accent: '#0891b2', tint: '#ecfeff' },
    Activation:       { accent: '#8b5bb5', tint: '#faf5ff' },
    Normalization:    { accent: '#b7791f', tint: '#fffbeb' },
    Regularization:   { accent: '#c06a37', tint: '#fff7ed' },
    Attention:        { accent: '#6366d3', tint: '#eef2ff' },
    Recurrent:        { accent: '#ab5a78', tint: '#fff1f2' },
    'Element-wise':   { accent: '#9b8426', tint: '#fefce8' },
    Combine:          { accent: '#72716d', tint: '#f5f5f4' },
};

const FALLBACK = { accent: '#64748b', tint: '#f1f5f9' };

export function styleFor(category) {
    return { card: 'bg-white text-slate-800', pill: 'bg-slate-100 text-slate-600', ...(STYLES[category] ?? FALLBACK) };
}
