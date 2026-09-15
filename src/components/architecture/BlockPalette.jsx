import React, { useMemo } from 'react';
import { Search } from 'lucide-react';
import { PYTORCH_BLOCKS, getEntry, listCategories } from '../../domain/library/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { styleFor } from './categoryStyle.js';

const TEMPLATE_ONLY = new Set(['deepflow.template.Input', 'deepflow.template.Output']);

const FAMILY_ORDER = [
    'IO',
    'Linear',
    'Convolution',
    'Pooling',
    'Activation',
    'Normalization',
    'Reshape',
    'Regularization',
    'Attention',
    'Recurrent',
    'Element-wise',
    'Combine',
];

const FAMILY_META = {
    IO:             { label: 'IO',    defaultKey: 'deepflow.io.Input' },
    Linear:         { label: 'Dense', defaultKey: 'torch.nn.Linear' },
    Convolution:    { label: 'Conv',  defaultKey: 'torch.nn.Conv2d' },
    Pooling:        { label: 'Pool',  defaultKey: 'torch.nn.MaxPool2d' },
    Activation:     { label: 'Act',   defaultKey: 'torch.nn.ReLU' },
    Normalization:  { label: 'Norm',  defaultKey: 'torch.nn.BatchNorm2d' },
    Reshape:        { label: 'Shape', defaultKey: 'torch.nn.Flatten' },
    Regularization: { label: 'Reg',   defaultKey: 'torch.nn.Dropout' },
    Attention:      { label: 'Attn',  defaultKey: 'torch.nn.MultiheadAttention' },
    Recurrent:      { label: 'RNN',   defaultKey: 'torch.nn.LSTM' },
    'Element-wise': { label: 'Math',  defaultKey: 'deepflow.op.Add' },
    Combine:        { label: 'Merge', defaultKey: 'deepflow.op.Concat' },
};

function matches(entry, query) {
    if (!query) return true;
    const needle = query.toLowerCase();
    return (
        entry.name.toLowerCase().includes(needle) ||
        entry.category.toLowerCase().includes(needle) ||
        entry.description.toLowerCase().includes(needle) ||
        entry.key.toLowerCase().includes(needle)
    );
}

const addPosition = () => ({
    x: 220 + Math.random() * 40,
    y: 220 + Math.random() * 80,
});

const Dot = ({ category }) => {
    const style = styleFor(category);
    return (
        <span
            className="inline-block h-2 w-2 rounded-full border border-black/25 shrink-0"
            style={{ backgroundColor: style.accent }}
        />
    );
};

const FamilyTile = ({ group }) => {
    const meta = FAMILY_META[group.category] ?? {
        label: group.category,
        defaultKey: group.items[0]?.key,
    };
    const entry = getEntry(meta.defaultKey) ?? group.items[0];
    const addLayerByLibraryKey = useArchitectureStore((s) => s.addLayerByLibraryKey);

    return (
        <button
            draggable={!!entry}
            onDragStart={(event) => {
                if (!entry) return;
                event.dataTransfer.setData('application/deepflow-lib-key', entry.key);
                event.dataTransfer.effectAllowed = 'move';
            }}
            onClick={() => entry && addLayerByLibraryKey(entry.key, addPosition())}
            title={entry ? `Add ${meta.label}; pick the exact module inside the node` : group.category}
            className="inline-flex h-7 min-w-0 items-center gap-1.5 rounded-md border border-black/12 bg-white px-2 transition-colors hover:border-black/35 hover:bg-stone-50"
        >
            <Dot category={group.category} />
            <span className="truncate text-[10px] font-bold uppercase tracking-wide">{meta.label}</span>
        </button>
    );
};

const SearchResult = ({ entry }) => {
    const addLayerByLibraryKey = useArchitectureStore((s) => s.addLayerByLibraryKey);
    return (
        <button
            draggable
            onDragStart={(event) => {
                event.dataTransfer.setData('application/deepflow-lib-key', entry.key);
                event.dataTransfer.effectAllowed = 'move';
            }}
            onClick={() => addLayerByLibraryKey(entry.key, addPosition())}
            title={entry.description}
            className="min-w-0 h-7 rounded-md border border-black/12 bg-white px-2 text-left transition-colors hover:border-black/35 hover:bg-stone-50 flex items-center gap-1.5"
        >
            <Dot category={entry.category} />
            <span className="truncate text-[11px] font-semibold">{entry.name}</span>
        </button>
    );
};

const BlockPalette = () => {
    const [query, setQuery] = React.useState('');

    const grouped = useMemo(() => {
        const categories = listCategories();
        const ordered = [
            ...FAMILY_ORDER.filter((category) => categories.includes(category)),
            ...categories.filter((category) => !FAMILY_ORDER.includes(category)),
        ];
        return ordered
            .map((category) => ({
                category,
                items: PYTORCH_BLOCKS.filter(
                    (entry) => entry.category === category && !TEMPLATE_ONLY.has(entry.key) && matches(entry, query),
                ),
            }))
            .filter((group) => group.items.length > 0);
    }, [query]);

    const searchEntries = grouped.flatMap((group) => group.items);

    return (
        <aside
            className="df-panel-subtle absolute pointer-events-auto overflow-hidden"
            style={{
                right: '1rem',
                top: '1rem',
                width: '208px',
                maxHeight: 'calc(100% - 2rem)',
                zIndex: 40,
            }}
        >
            {/* Section header */}
            <div className="flex items-center justify-between px-3 pt-2.5 pb-1.5">
                <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-black/80">
                    Blocks
                </div>
            </div>

            <div className="px-2.5 pb-2">
                <div className="df-input flex h-7 items-center gap-2 px-2">
                    <Search size={12} className="shrink-0 opacity-70" />
                    <input
                        type="text"
                        placeholder="Search..."
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        className="min-w-0 flex-1 border-none bg-transparent text-[11px] outline-none focus:ring-0 placeholder:text-gray-500"
                    />
                </div>
            </div>

            <div className="overflow-y-auto px-2.5 pb-2.5" style={{ maxHeight: 'calc(100vh - 8rem)' }}>
                {query ? (
                    <div className="grid grid-cols-2 gap-1">
                        {searchEntries.map((entry) => (
                            <SearchResult key={entry.key} entry={entry} />
                        ))}
                    </div>
                ) : (
                    <div className="grid grid-cols-2 gap-1">
                        {grouped.map((group) => (
                            <FamilyTile key={group.category} group={group} />
                        ))}
                    </div>
                )}
                {grouped.length === 0 && (
                    <div className="text-xs italic text-text-secondary">No matches for "{query}".</div>
                )}
            </div>
        </aside>
    );
};

export default BlockPalette;
