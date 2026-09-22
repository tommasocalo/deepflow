/**
 * Category → visual style. Keep these centralized so the palette and node
 * cards stay in sync.
 */
const STYLES = {
    IO:               { card: 'bg-gray-900 text-white',   pill: 'bg-gray-900 text-white',   accent: '#111827' },
    Convolution:      { card: 'bg-emerald-100',           pill: 'bg-emerald-200',           accent: '#10b981' },
    Pooling:          { card: 'bg-green-100',             pill: 'bg-green-200',             accent: '#22c55e' },
    Linear:           { card: 'bg-blue-100',              pill: 'bg-blue-200',              accent: '#3b82f6' },
    Reshape:          { card: 'bg-cyan-100',              pill: 'bg-cyan-200',              accent: '#06b6d4' },
    Activation:       { card: 'bg-violet-100',            pill: 'bg-violet-200',            accent: '#8b5cf6' },
    Normalization:    { card: 'bg-amber-100',             pill: 'bg-amber-200',             accent: '#f59e0b' },
    Regularization:   { card: 'bg-orange-100',            pill: 'bg-orange-200',            accent: '#f97316' },
    Attention:        { card: 'bg-pink-100',              pill: 'bg-pink-200',              accent: '#ec4899' },
    Recurrent:        { card: 'bg-rose-100',              pill: 'bg-rose-200',              accent: '#f43f5e' },
    'Element-wise':   { card: 'bg-yellow-100',            pill: 'bg-yellow-200',            accent: '#eab308' },
    Combine:          { card: 'bg-stone-200',             pill: 'bg-stone-300',             accent: '#78716c' },
    Parameters:       { card: 'bg-amber-50',              pill: 'bg-amber-100',             accent: '#a16207' },
};

const FALLBACK = { card: 'bg-white', pill: 'bg-gray-200', accent: '#000000' };

export function styleFor(category) {
    return STYLES[category] ?? FALLBACK;
}
