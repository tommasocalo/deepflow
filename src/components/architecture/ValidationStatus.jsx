import React, { useState } from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp } from 'lucide-react';
import { useValidation } from '../../hooks/useValidation.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';

/**
 * Compact status pill that lives in the tab strip. Click to expand a
 * dropdown of all errors/warnings; clicking an entry selects the offending
 * node so the inspector opens to it.
 */
const ValidationStatus = () => {
    const validation = useValidation();
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const [open, setOpen] = useState(false);

    const errorCount = Object.values(validation.errors).reduce((a, e) => a + e.length, 0)
        + validation.globalErrors.length;
    const warningCount = Object.values(validation.warnings).reduce((a, w) => a + w.length, 0);
    const ok = errorCount === 0 && warningCount === 0;

    const tone = errorCount > 0
        ? { bg: 'bg-red-100', text: 'text-red-700', icon: <AlertCircle size={12} strokeWidth={3} /> }
        : warningCount > 0
            ? { bg: 'bg-amber-100', text: 'text-amber-700', icon: <AlertTriangle size={12} strokeWidth={3} /> }
            : { bg: 'bg-emerald-100', text: 'text-emerald-700', icon: <CheckCircle2 size={12} strokeWidth={3} /> };

    const label = ok
        ? 'Graph valid'
        : `${errorCount} error${errorCount !== 1 ? 's' : ''}${warningCount ? `, ${warningCount} warning${warningCount !== 1 ? 's' : ''}` : ''}`;

    return (
        <div className="relative">
            <button
                onClick={() => !ok && setOpen((o) => !o)}
                className={`${tone.bg} ${tone.text} flex items-center gap-1 px-2 py-1 border-2 border-black rounded-md text-[11px] font-bold tracking-wide shadow-medium`}
                disabled={ok}
            >
                {tone.icon}
                {label}
                {!ok && (open ? <ChevronUp size={12} strokeWidth={3} /> : <ChevronDown size={12} strokeWidth={3} />)}
            </button>
            {open && !ok && (
                <div
                    className="absolute right-0 mt-1 w-80 max-h-80 overflow-y-auto bg-white border-2 border-black rounded-md shadow-hard z-50"
                    onMouseLeave={() => setOpen(false)}
                >
                    {validation.globalErrors.length > 0 && (
                        <div className="p-2 border-b-2 border-black bg-red-50">
                            <div className="text-[10px] font-bold uppercase tracking-wider text-red-700 mb-0.5">Graph</div>
                            <ul className="text-xs text-red-800 space-y-0.5 list-disc pl-4">
                                {validation.globalErrors.map((e, i) => <li key={i}>{e}</li>)}
                            </ul>
                        </div>
                    )}
                    {Object.entries(validation.errors).map(([id, msgs]) => (
                        <button
                            key={id}
                            onClick={() => { selectOnly(id); setOpen(false); }}
                            className="w-full text-left p-2 border-b border-black/20 hover:bg-red-50"
                        >
                            <div className="text-[10px] font-bold uppercase tracking-wider text-red-700 mb-0.5">Node error</div>
                            <ul className="text-xs space-y-0.5 list-disc pl-4">
                                {msgs.map((m, i) => <li key={i}>{m}</li>)}
                            </ul>
                        </button>
                    ))}
                    {Object.entries(validation.warnings).map(([id, msgs]) => (
                        <button
                            key={id}
                            onClick={() => { selectOnly(id); setOpen(false); }}
                            className="w-full text-left p-2 border-b border-black/20 hover:bg-amber-50"
                        >
                            <div className="text-[10px] font-bold uppercase tracking-wider text-amber-700 mb-0.5">Warning</div>
                            <ul className="text-xs space-y-0.5 list-disc pl-4">
                                {msgs.map((m, i) => <li key={i}>{m}</li>)}
                            </ul>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
};

export default ValidationStatus;
