import { useId, useState } from 'react';
import { useBridgeSource } from '../useBridgeSource.js';
import { ClassSwatch } from './ClassMarker.jsx';
import ScatterPreview from './ScatterPreview.jsx';
import { dataSeedOf, datasetSize, parseInputShape, resolveOptions, SPLIT_LIMITS } from './registry.js';
import { splitSizes } from './split.js';
import { useEduDataStore } from './store.js';
import { dtx } from './text.js';
import { applyDataset } from './useDataset.js';
import { usePrepared } from './usePrepared.js';

const formatInt = (value) => (Number.isFinite(value) ? value.toLocaleString('en-US') : '?');
const percent = (fraction) => `${Math.round(fraction * 100)}%`;

/** "150 flowers · 4 features · 3 classes" */
function summaryLine(spec, n) {
    const dims = parseInputShape(spec.input.shape) ?? [];
    const parts = [];
    if (spec.kind === 'toy2d') parts.push(dtx('summary.points', { n: formatInt(n) }));
    else if (spec.kind === 'tabular') parts.push(dtx('summary.rows', { n: formatInt(n), unit: dtx(spec.id === 'iris' ? 'unit.iris' : 'unit.default'), features: dims[0] }));
    else if (spec.kind === 'image') parts.push(dtx('summary.images', { n: formatInt(n), shape: dims.join(' × ') }));
    else parts.push(dtx('summary.text', { n: formatInt(spec.size?.total ?? n) }));
    if (spec.target.kind === 'classes') parts.push(dtx('summary.classes', { count: spec.target.size }));
    else if (spec.target.kind === 'regression') parts.push(dtx('summary.regression'));
    else parts.push(dtx('summary.vocab', { size: spec.target.size }));
    return parts.join(' · ');
}

function outputText(spec) {
    if (spec.target.kind === 'regression') return dtx('use.outputRegression');
    if (spec.target.kind === 'next-token') return dtx('use.outputTokens', { size: spec.target.size });
    if (spec.target.size === 2) return dtx('use.outputBinary');
    return dtx('use.outputClasses', { count: spec.target.size });
}

function ioLine(spec) {
    const output = spec.target.kind === 'classes' ? dtx('io.outputClasses', { count: spec.target.size })
        : spec.target.kind === 'regression' ? dtx('io.outputRegression') : dtx('io.outputTokens', { size: spec.target.size });
    return `${dtx('io.input', { shape: spec.input.shape })} · ${spec.input.dtype} · ${output}`;
}

function UseStatus({ spec, boundId, lastUse }) {
    if (lastUse && lastUse.datasetId === spec.id) {
        return (
            <>
                {lastUse.changed.length > 0 && <span>{dtx('use.result', { changes: lastUse.changed.join('; ') })}</span>}
                {lastUse.warnings.map((warning) => <span key={warning} className="edu-data-warning">{dtx('use.warning', { warning })}</span>)}
            </>
        );
    }
    if (boundId === spec.id) return <span>{dtx('use.inUse')}</span>;
    if (boundId) return <span>{dtx('use.other', { name: useEduDataStore.getState().current().spec?.name ?? boundId })}</span>;
    return <span>{dtx('use.none')}</span>;
}

function DigitThumbnails({ prepared }) {
    const { raw } = prepared;
    const picks = [];
    for (let label = 0; label < 10; label++) {
        let found = 0;
        for (let i = 0; i < raw.n && found < 2; i++) {
            if (raw.y[i] === label) { picks.push(i); found += 1; }
        }
    }
    return (
        <figure className="edu-data-digits">
            <ul className="edu-data-digit-grid">
                {picks.map((index) => (
                    <li key={index}>
                        <svg viewBox="0 0 8 8" width="40" height="40" role="img" aria-label={dtx('preview.digitLabel', { label: raw.y[index] })} shapeRendering="crispEdges">
                            {Array.from({ length: 64 }, (_, pixel) => {
                                const ink = raw.x[index * 64 + pixel] / 16;
                                if (ink <= 0) return null;
                                const shade = Math.round(255 - ink * 235);
                                return <rect key={pixel} x={pixel % 8} y={Math.floor(pixel / 8)} width="1" height="1" fill={`rgb(${shade}, ${shade}, ${shade})`} />;
                            })}
                        </svg>
                        <span aria-hidden="true">{raw.y[index]}</span>
                    </li>
                ))}
            </ul>
            <figcaption className="edu-data-note">{dtx('preview.digitsTitle')}</figcaption>
        </figure>
    );
}

function RowsTable({ prepared }) {
    const { raw } = prepared;
    const names = prepared.featureNames ?? [];
    const role = new Map();
    for (const index of prepared.val.indices) role.set(index, 'val');
    for (const index of prepared.test.indices) role.set(index, 'test');
    const count = Math.min(8, raw.n);
    return (
        <div className="edu-data-table-wrap">
            <table className="edu-data-table">
                <caption className="edu-data-note">{dtx('preview.tableCaption', { count, n: raw.n })}</caption>
                <thead>
                    <tr>
                        {names.map((name) => <th key={name} scope="col">{name}</th>)}
                        <th scope="col">{dtx('classes.target')}</th>
                        <th scope="col">{dtx('preview.split')}</th>
                    </tr>
                </thead>
                <tbody>
                    {Array.from({ length: count }, (_, row) => (
                        <tr key={row}>
                            {names.map((name, f) => <td key={name} className="edu-data-num">{raw.x[row * prepared.featureSize + f].toFixed(1)}</td>)}
                            <td>{prepared.classNames ? prepared.classNames[raw.y[row]] : raw.y[row].toFixed(3)}</td>
                            <td>{role.get(row) ?? 'train'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

function Legend({ prepared }) {
    return (
        <div className="edu-data-legend">
            {prepared.classNames && (
                <ul className="edu-data-legend-classes">
                    {prepared.classNames.map((name, index) => (
                        <li key={name}><ClassSwatch index={index} /> {name}</li>
                    ))}
                </ul>
            )}
            <p className="edu-data-note">
                {dtx('preview.legendSplit')}{prepared.test.n ? ` · ${dtx('preview.legendTest')}` : ''}
            </p>
        </div>
    );
}

function Preview({ spec, prepared, loading, error }) {
    const [mode, setMode] = useState('scatter');
    const features = prepared?.featureNames?.length ?? 0;
    const [axes, setAxes] = useState(() => (spec.id === 'iris' ? [2, 3] : [0, 1]));
    const headingId = useId();
    let body;
    if (spec.training !== 'browser') {
        body = (
            <div className="edu-data-schematic">
                {spec.schematic && <p>{spec.schematic}</p>}
                <p className="edu-data-note">{dtx('preview.none')}</p>
            </div>
        );
    } else if (error) {
        body = <p className="edu-data-warning">{dtx('preview.loadFailed', { message: error })}</p>;
    } else if (!prepared) {
        body = <p className="edu-data-note" role="status">{loading ? dtx('preview.loading') : ''}</p>;
    } else if (spec.preview === 'images') {
        body = <DigitThumbnails prepared={prepared} />;
    } else if (mode === 'table' && spec.preview === 'tabular') {
        body = <RowsTable prepared={prepared} />;
    } else {
        body = (
            <>
                {features > 2 && (
                    <div className="edu-data-axis-pickers">
                        {[0, 1].map((axis) => (
                            <label key={axis}>
                                <span>{dtx(axis === 0 ? 'preview.x' : 'preview.y')}</span>
                                <select className="df-input edu-data-select" value={axes[axis]}
                                    onChange={(event) => setAxes((current) => (axis === 0 ? [Number(event.target.value), current[1]] : [current[0], Number(event.target.value)]))}>
                                    {prepared.featureNames.map((name, index) => <option key={name} value={index}>{name}</option>)}
                                </select>
                            </label>
                        ))}
                    </div>
                )}
                <ScatterPreview prepared={prepared} xFeature={features > 2 ? axes[0] : 0} yFeature={features > 2 ? axes[1] : 1} name={spec.name} />
                <Legend prepared={prepared} />
            </>
        );
    }
    return (
        <section className="edu-data-card edu-data-preview" data-edu-target="dataset-preview" aria-labelledby={headingId}>
            <div className="edu-data-card-head">
                <h3 id={headingId}>{dtx('preview.title')}</h3>
                {spec.preview === 'tabular' && prepared && (
                    <div className="edu-data-segmented" role="group" aria-label={dtx('preview.mode')}>
                        {['scatter', 'table'].map((value) => (
                            <button key={value} type="button" aria-pressed={mode === value} className={mode === value ? 'is-on' : ''} onClick={() => setMode(value)}>
                                {dtx(`preview.${value}`)}
                            </button>
                        ))}
                    </div>
                )}
            </div>
            {body}
        </section>
    );
}

function ClassBalance({ spec, prepared }) {
    const headingId = useId();
    let body;
    if (spec.target.kind === 'regression') body = <p className="edu-data-note">{dtx('classes.regression')}</p>;
    else if (spec.target.kind === 'next-token') body = <p className="edu-data-note">{dtx('classes.nextToken', { size: spec.target.size })}</p>;
    else if (!prepared) {
        body = (
            <ul className="edu-data-chips">
                {spec.target.classes.map((name) => <li key={name}>{name}</li>)}
            </ul>
        );
    } else {
        const count = (split, index) => { let c = 0; for (let i = 0; i < split.n; i++) if (split.y[i] === index) c += 1; return c; };
        const rows = prepared.classNames.map((name, index) => {
            const train = count(prepared.train, index), val = count(prepared.val, index), test = count(prepared.test, index);
            return { name, index, train, val, test, total: train + val + test };
        });
        const max = Math.max(1, ...rows.map((row) => row.total));
        const compact = rows.length > 4;
        const totals = { train: formatInt(prepared.train.n), val: formatInt(prepared.val.n), test: formatInt(prepared.test.n) };
        body = (
            <>
                <ul className={`edu-data-balance${compact ? ' is-compact' : ''}`}>
                    {rows.map((row) => (
                        <li key={row.name}>
                            <span className="edu-data-balance-name">{!compact && <ClassSwatch index={row.index} />}{row.name}</span>
                            <span className="edu-data-balance-bar" aria-hidden="true"><span style={{ width: `${(row.total / max) * 100}%` }} /></span>
                            <span className="edu-data-balance-count">{formatInt(row.total)}</span>
                            {!compact && (
                                <span className="edu-data-balance-split edu-data-note">
                                    {prepared.test.n ? dtx('classes.splitTest', { train: row.train, val: row.val, test: row.test }) : dtx('classes.split', { train: row.train, val: row.val })}
                                </span>
                            )}
                        </li>
                    ))}
                </ul>
                {compact && <p className="edu-data-note">{dtx(prepared.test.n ? 'classes.totalTest' : 'classes.total', totals)}</p>}
            </>
        );
    }
    return (
        <section className="edu-data-card" aria-labelledby={headingId}>
            <div className="edu-data-card-head"><h3 id={headingId}>{spec.target.kind === 'classes' ? dtx('classes.title') : dtx('classes.target')}</h3></div>
            {body}
        </section>
    );
}

function NumberOption({ name, option, value, onChange }) {
    const range = name !== 'seed';
    const display = name === 'noise' ? value.toFixed(2) : String(value);
    return (
        <label className={`edu-data-option${range ? ' is-range' : ''}`}>
            <span className="edu-data-option-label">{option.label}</span>
            {range ? (
                <input type="range" min={option.min} max={option.max} step={option.step ?? 1} value={value}
                    onChange={(event) => onChange(Number(event.target.value))} />
            ) : (
                <input className="df-input edu-data-number" type="number" min={option.min} max={option.max} step={1} value={value}
                    onChange={(event) => { if (event.target.value !== '') onChange(Number(event.target.value)); }} />
            )}
            {range && <output className="edu-data-option-value">{display}</output>}
        </label>
    );
}

function Options({ spec, options, split }) {
    const headingId = useId();
    const store = useEduDataStore.getState();
    const n = datasetSize(spec, options);
    const sizes = splitSizes(n, split);
    const entries = Object.entries(spec.options ?? {});
    return (
        <section className="edu-data-card" data-edu-target="dataset-options" aria-labelledby={headingId}>
            <div className="edu-data-card-head"><h3 id={headingId}>{dtx('options.title')}</h3></div>
            {spec.training === 'browser' ? (
                <>
                    <div className="edu-data-option-grid">
                        {entries.map(([name, option]) => (
                            <NumberOption key={name} name={name} option={option} value={options[name]} onChange={(value) => store.setOptions({ [name]: value }, spec.id)} />
                        ))}
                    </div>
                    <p className="edu-data-note">{dtx(spec.kind === 'toy2d' ? 'options.seedHelp' : 'options.seedHelpSplit')}</p>
                    <div className="edu-data-option-grid">
                        <label className="edu-data-option is-range">
                            <span className="edu-data-option-label">{dtx('options.val')}</span>
                            <input type="range" min={SPLIT_LIMITS.val.min} max={SPLIT_LIMITS.val.max} step={SPLIT_LIMITS.val.step} value={split.val}
                                onChange={(event) => store.setSplit({ val: Number(event.target.value) })} />
                            <output className="edu-data-option-value">{percent(split.val)}</output>
                        </label>
                        <label className="edu-data-option">
                            <span className="edu-data-option-label">{dtx('options.test')}</span>
                            <select className="df-input edu-data-select" value={split.test} onChange={(event) => store.setSplit({ test: Number(event.target.value) })}>
                                {SPLIT_LIMITS.test.map((value) => <option key={value} value={value}>{percent(value)}</option>)}
                            </select>
                        </label>
                    </div>
                    <p className="edu-data-split-sizes">{dtx('options.sizes', { train: formatInt(sizes.nTrain), val: formatInt(sizes.nVal), test: formatInt(sizes.nTest) })}</p>
                </>
            ) : (
                <p className="edu-data-note">
                    {spec.size?.train && spec.size?.test
                        ? dtx('options.officialSplit', { train: formatInt(spec.size.train), test: formatInt(spec.size.test), val: formatInt(spec.valSize ?? spec.options?.valSize?.default ?? 5000) })
                        : dtx('options.exportSplit')}
                </p>
            )}
        </section>
    );
}

function Scaling({ spec, prepared }) {
    const headingId = useId();
    let body;
    if (spec.training !== 'browser') {
        const norm = spec.normalization;
        body = norm
            ? <p>{dtx('scaling.normalize', { mean: `(${norm.mean.join(', ')})`, std: `(${norm.std.join(', ')})` })}</p>
            : <p className="edu-data-note">{dtx('options.exportSplit')}</p>;
    } else if (spec.scaling === 'standardize') {
        body = (
            <>
                <p>{renderBold(dtx('scaling.standardize', { rows: prepared ? prepared.train.n : '…' }))}</p>
                {prepared?.scaling && (
                    <div className="edu-data-table-wrap">
                        <table className="edu-data-table edu-data-table-compact">
                            <thead><tr><th scope="col">{dtx('scaling.feature')}</th><th scope="col">{dtx('scaling.mean')}</th><th scope="col">{dtx('scaling.std')}</th></tr></thead>
                            <tbody>
                                {prepared.featureNames.map((name, f) => (
                                    <tr key={name}><th scope="row">{name}</th><td className="edu-data-num">{prepared.scaling.mean[f].toFixed(2)}</td><td className="edu-data-num">{prepared.scaling.std[f].toFixed(2)}</td></tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </>
        );
    } else {
        body = <p>{dtx(spec.scaling === 'divide16' ? 'scaling.divide16' : 'scaling.none')}</p>;
    }
    return (
        <section className="edu-data-card" aria-labelledby={headingId}>
            <div className="edu-data-card-head"><h3 id={headingId}>{dtx('scaling.title')}</h3></div>
            {body}
        </section>
    );
}

/** **bold** spans only (copy written by us, never user input). */
function renderBold(text) {
    return text.split('**').map((part, index) => (index % 2 ? <strong key={index}>{part}</strong> : part));
}

function Source({ spec }) {
    const headingId = useId();
    const licence = spec.licence ?? {};
    return (
        <section className="edu-data-card" aria-labelledby={headingId}>
            <div className="edu-data-card-head"><h3 id={headingId}>{dtx('source.title')}</h3></div>
            <p>{spec.attribution}</p>
            <p className="edu-data-note">
                {dtx('source.licence')}: {licence.url ? <a href={licence.url} target="_blank" rel="noreferrer">{licence.name}</a> : licence.name}
                {licence.verified === false ? ` (${dtx('source.unverified')})` : ''}
                {licence.note ? `. ${licence.note}` : ''}
            </p>
            {spec.citation?.length > 0 && (
                <details className="edu-data-cite">
                    <summary>{dtx('source.cite')}</summary>
                    <ul>{spec.citation.map((line) => <li key={line}>{line}</li>)}</ul>
                </details>
            )}
        </section>
    );
}

/** The detail pane of one dataset (docs/edu/SPEC.md §3.4). */
export default function DatasetDetail({ spec }) {
    const boundId = useEduDataStore((state) => state.datasetId);
    const stored = useEduDataStore((state) => state.options[spec.id]);
    const split = useEduDataStore((state) => state.split);
    const lastUse = useEduDataStore((state) => state.lastUse);
    const options = resolveOptions(spec, stored ?? {});
    const { prepared, loading, error } = usePrepared(spec, options, split, dataSeedOf(options));
    const n = datasetSize(spec, options);
    const headingId = useId();
    // In a lesson step where choosing the output size is the exercise, the button sets only the Input (§3.4).
    const lessonUse = useBridgeSource('lesson')?.datasetUse ?? null;
    const inputOnly = lessonUse?.setOutput === false;
    return (
        <section className="edu-data-detail" aria-labelledby={headingId}>
            <header className="edu-data-header">
                <div className="edu-data-titlebar">
                    <h2 id={headingId} className="edu-data-title">
                        {spec.name} <span className="edu-data-summary">— {summaryLine(spec, n)}</span>
                    </h2>
                    <span className={`edu-data-badge is-${spec.training}`}>{dtx(spec.training === 'browser' ? 'badge.browser' : 'badge.export')}</span>
                </div>
                <p className="edu-data-desc">{spec.description}</p>
                <div className="edu-data-use">
                    <button type="button" className="df-button df-button-primary edu-data-use-button" data-edu-target="use-dataset" onClick={() => applyDataset(spec.id, lessonUse ?? {})}>
                        {dtx('use.button')}
                    </button>
                    <span className="edu-data-io">{ioLine(spec)}</span>
                </div>
                <p className="edu-data-note edu-data-use-hint">
                    {inputOnly ? dtx('use.hintInputOnly', { shape: spec.input.shape }) : dtx('use.hint', { shape: spec.input.shape, output: outputText(spec) })}
                </p>
                <p className="edu-data-use-status" role="status"><UseStatus spec={spec} boundId={boundId} lastUse={lastUse} /></p>
            </header>
            <div className="edu-data-grid">
                <Preview key={spec.id} spec={spec} prepared={prepared} loading={loading} error={error} />
                <div className="edu-data-side">
                    <ClassBalance spec={spec} prepared={prepared} />
                    <Options spec={spec} options={options} split={split} />
                    <Scaling spec={spec} prepared={prepared} />
                    <Source spec={spec} />
                </div>
            </div>
        </section>
    );
}
