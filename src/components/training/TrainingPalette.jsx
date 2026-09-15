import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Cog, Layers3, Target } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore.js';

const OPTIMIZERS = [
    { type: 'Adam',    label: 'Adam',    desc: 'Adaptive moment estimation', showcase: true },
    { type: 'AdamW',   label: 'AdamW',   desc: 'Adam with decoupled weight decay' },
    { type: 'SGD',     label: 'SGD',     desc: 'Stochastic gradient descent', showcase: true },
    { type: 'RMSprop', label: 'RMSprop', desc: 'Root mean square propagation' },
];

const LOSSES = [
    { type: 'cross_entropy', label: 'Cross-Entropy', desc: 'Multi-class classification', showcase: true },
    { type: 'bce',           label: 'BCE',           desc: 'Binary classification', showcase: true },
    { type: 'mse',           label: 'MSE',           desc: 'Regression' },
    { type: 'l1',            label: 'L1',            desc: 'Robust regression' },
    { type: 'custom',        label: 'Custom',        desc: 'User-defined loss module' },
];

const Row = ({ icon: Icon, accent, label, desc, onDragStart, onClick }) => (
    <button
        draggable
        onDragStart={onDragStart}
        onClick={onClick}
        title={desc}
        className="w-full flex items-center gap-2.5 px-3 h-7 text-xs text-left hover:bg-stone-100 transition-colors"
    >
        <span
            className="inline-block w-2 h-2 rounded-full border border-black/30 shrink-0"
            style={{ backgroundColor: accent }}
        />
        <Icon size={11} strokeWidth={2.5} className="opacity-60 shrink-0" />
        <span className="truncate">{label}</span>
    </button>
);

const Category = ({ name, items, renderRow }) => {
    const [open, setOpen] = useState(false);
    return (
        <div className="border-b border-black/10 last:border-b-0">
            <button
                onClick={() => setOpen((v) => !v)}
                className="w-full flex items-center gap-1.5 px-3 h-8 text-[10px] font-bold uppercase tracking-wider text-text-secondary hover:bg-stone-50"
            >
                {open
                    ? <ChevronDown size={11} strokeWidth={3} />
                    : <ChevronRight size={11} strokeWidth={3} />
                }
                <span className="flex-1 text-left">{name}</span>
            </button>
            {open && (
                <div className="px-1.5 pb-1.5">{items.map(renderRow)}</div>
            )}
        </div>
    );
};

const TrainingPalette = () => {
    const addOptimizer = useTrainingStore((s) => s.addOptimizer);
    const addLoss = useTrainingStore((s) => s.addLoss);
    const addPhase = useTrainingStore((s) => s.addPhase);

    const renderOpt = (o) => (
        <Row
            key={o.type}
            icon={Cog}
            accent="#bfdbfe"
            label={o.label}
            desc={o.desc}
            onDragStart={(e) => {
                e.dataTransfer.setData('application/deepflow-optimizer', o.type);
                e.dataTransfer.effectAllowed = 'move';
            }}
            onClick={() => addOptimizer(o.type, { x: 200 + Math.random() * 60, y: 100 + Math.random() * 80 })}
        />
    );
    const renderLoss = (l) => (
        <Row
            key={l.type}
            icon={Target}
            accent="#fbcfe8"
            label={l.label}
            desc={l.desc}
            onDragStart={(e) => {
                e.dataTransfer.setData('application/deepflow-loss', l.type);
                e.dataTransfer.effectAllowed = 'move';
            }}
            onClick={() => addLoss(l.type, { x: 200 + Math.random() * 60, y: 320 + Math.random() * 80 })}
        />
    );
    const phaseItem = { showcase: true, render: () => (
        <Row
            key="phase"
            icon={Layers3}
            accent="#a7f3d0"
            label="Phase"
            desc="Schedule a sub-step (G/D, EMA, curriculum)"
            onDragStart={(e) => {
                e.dataTransfer.setData('application/deepflow-phase', 'Phase');
                e.dataTransfer.effectAllowed = 'move';
            }}
            onClick={() => addPhase({ x: 600 + Math.random() * 60, y: 220 + Math.random() * 80 })}
        />
    ) };

    return (
        <div
            className="df-panel-subtle absolute pointer-events-auto overflow-hidden"
            style={{ right: '1rem', top: '1rem', width: '240px', maxHeight: 'calc(100% - 2rem)', zIndex: 40 }}
        >
            <div className="px-3 py-2 border-b border-black/15 bg-white shrink-0">
                <div className="flex items-center justify-between px-1">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-black">
                        Training
                    </div>
                    <div className="text-[9px] font-mono text-text-secondary">blocks</div>
                </div>
            </div>
            <div className="overflow-y-auto" style={{ maxHeight: 'calc(100vh - 8.5rem)' }}>
                <div className="border-b border-black/15 bg-white">
                    <div className="px-3 pt-2 pb-1 text-[10px] font-bold uppercase tracking-wider text-black">
                        Frequently Used
                    </div>
                    <div className="px-1.5 pb-2">
                        {OPTIMIZERS.filter((it) => it.showcase).map(renderOpt)}
                        {LOSSES.filter((it) => it.showcase).map(renderLoss)}
                    </div>
                </div>
                <Category name="Optimizers" items={OPTIMIZERS} renderRow={renderOpt} />
                <Category name="Losses"     items={LOSSES}     renderRow={renderLoss} />
                <Category name="Schedule"   items={[phaseItem]} renderRow={(it) => it.render()} />
            </div>
        </div>
    );
};

export default TrainingPalette;
