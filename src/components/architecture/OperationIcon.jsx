import React from 'react';
import { getEntry } from '../../domain/library/index.js';

const FAMILY_ASSETS = {
    Linear: 'linear', Attention: 'attention', Convolution: 'convolution',
    Normalization: 'normalization', Activation: 'activation', Pooling: 'pooling',
    Regularization: 'dropout', Recurrent: 'recurrent',
};

/** One visual vocabulary, shared by the canvas and the block library. */
export default function OperationIcon({ libraryKey, size = 32, className = '' }) {
    const asset = FAMILY_ASSETS[getEntry(libraryKey)?.category];
    if (asset) return <img className={`df-operation-icon ${className}`} src={`/assets/operations/${asset}.png`} width={size} height={size} alt="" draggable={false} />;

    let mark;
    switch (libraryKey) {
        case 'deepflow.op.Add': mark = <path d="M16 7v18M7 16h18" />; break;
        case 'deepflow.op.Multiply': mark = <path d="m9 9 14 14M23 9 9 23" />; break;
        case 'deepflow.op.MatMul': mark = <><path d="M8 4H4v24h4m16-24h4v24h-4M11 11l10 10m0-10L11 21" /></>; break;
        case 'deepflow.op.Transpose': mark = <><rect x="5" y="5" width="8" height="20" /><path d="M18 6h8v8m0-8-8 8M19 21h8v5H17" /></>; break;
        case 'deepflow.op.Scale': mark = <><path d="m6 7 9 9m0-9-9 9M24 5v22M20 27h8" /><path d="M5 23h10" /></>; break;
        case 'deepflow.op.CausalMask': mark = <><rect x="5" y="5" width="22" height="22" /><path d="M5 12h7v7h7v8M5 19h7" /><path d="m17 7 8 8m-8-3 8 8" opacity=".4" /></>; break;
        case 'deepflow.op.Broadcast': mark = <><path d="M16 4v10M5 14h22M5 14v13m11-13v13m11-13v13m-25-4 3 4 3-4m5 0 3 4 3-4m5 0 3 4 3-4" /></>; break;
        case 'deepflow.op.Stack': mark = <><path d="M5 6h18v14H5zM9 24h18V10M13 28h18V14" /></>; break;
        case 'deepflow.op.Concat': mark = <><path d="M6 5v5l10 6 10-6V5M16 16v11" /><path d="m12 23 4 4 4-4" /></>; break;
        case 'deepflow.op.Split': mark = <><path d="M16 5v10L6 21v6m10-12 10 6v6" /><path d="m3 24 3 3 3-3m14 0 3 3 3-3" /></>; break;
        case 'torch.nn.Flatten': mark = <><rect x="4" y="4" width="10" height="10" /><path d="M19 9h7m-3-3 3 3-3 3M5 24h22M5 21v6m7-6v6m8-6v6m7-6v6" /></>; break;
        case 'deepflow.io.Input':
        case 'deepflow.template.Input': mark = <><path d="M16 4v16m-6-6 6 6 6-6M6 22v6h20v-6" /></>; break;
        case 'deepflow.io.Output':
        case 'deepflow.template.Output': mark = <><path d="M16 20V4m-6 6 6-6 6 6M6 22v6h20v-6" /></>; break;
        default: mark = <><rect x="6" y="6" width="20" height="20" /><path d="M11 16h10M16 11v10" /></>;
    }
    return <svg className={`df-operation-icon ${className}`} width={size} height={size} viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="square" strokeLinejoin="miter" aria-hidden="true">{mark}</svg>;
}
