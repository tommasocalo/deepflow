import { browserDatasets, exportDatasets } from './registry.js';
import { dtx } from './text.js';

function metaLine(spec) {
    const kind = dtx(`meta.${spec.kind}`);
    const target = spec.target.kind === 'classes' ? dtx('meta.classes', { count: spec.target.size })
        : spec.target.kind === 'regression' ? dtx('meta.regression') : dtx('meta.nextToken');
    return `${kind} · ${target}`;
}

function Group({ title, specs, viewId, boundId, onSelect }) {
    if (!specs.length) return null;
    return (
        <section className="edu-data-group" aria-label={title}>
            <h3 className="edu-data-group-title">{title}</h3>
            <ul className="edu-data-rows">
                {specs.map((spec) => {
                    const bound = spec.id === boundId;
                    const viewed = spec.id === viewId;
                    return (
                        <li key={spec.id}>
                            <button
                                type="button"
                                className={`edu-data-row${viewed ? ' is-viewed' : ''}${bound ? ' is-bound' : ''}`}
                                data-edu-target={`dataset-${spec.id}`}
                                aria-current={viewed ? 'true' : undefined}
                                aria-label={bound ? dtx('list.inUseLabel', { name: spec.name }) : spec.name}
                                onClick={() => onSelect(spec.id)}
                            >
                                <span className={`edu-data-dot${bound ? ' is-on' : ''}`} aria-hidden="true" />
                                <span className="edu-data-row-text">
                                    <span className="edu-data-row-name">{spec.name}</span>
                                    <span className="edu-data-row-meta">{metaLine(spec)}</span>
                                </span>
                                {bound && <span className="edu-data-inuse" aria-hidden="true">{dtx('list.inUse')}</span>}
                            </button>
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}

/** The dataset list, grouped by where it trains (docs/edu/SPEC.md §3.4). Rows carry data-edu-target="dataset-<id>". */
export default function DatasetList({ viewId, boundId, onSelect }) {
    return (
        <nav className="edu-data-list" aria-label={dtx('list.title')}>
            <h2 className="edu-data-list-title">{dtx('list.title')}</h2>
            <Group title={dtx('list.browser')} specs={browserDatasets()} viewId={viewId} boundId={boundId} onSelect={onSelect} />
            <Group title={dtx('list.export')} specs={exportDatasets()} viewId={viewId} boundId={boundId} onSelect={onSelect} />
        </nav>
    );
}
