import { ChevronRight } from 'lucide-react';
import { t } from '../lessons/engine/i18n.js';
import { datasetName } from './format.js';

const STATUS_ICON = { 'not-started': '○', 'in-progress': '◐', completed: '●' };

/**
 * One module on the Learn home (docs/edu/SPEC.md §3.2): number and title, a status with a text label, the goal,
 * dataset chips, the duration and one button. Modules are never locked.
 */
export default function ModuleCard({ module, summary, recommendedAfter, onOpen }) {
    const number = module.order;
    const title = t(`${module.id}.title`);
    const status = summary.status;
    const action = status === 'completed' ? 'review' : status === 'in-progress' ? 'continue' : 'start';
    const statusLabel = status === 'completed' ? t('card.status.completed')
        : status === 'in-progress' ? t('card.status.inProgress', { done: summary.done, total: summary.total }) : t('card.status.notStarted');
    const statusShort = status === 'in-progress' ? t('card.status.inProgressShort', { done: summary.done, total: summary.total }) : statusLabel;
    const headingId = `edu-module-${module.id}`;
    return (
        <section className={`edu-module-card is-${status}`} aria-labelledby={headingId} data-module={module.id}>
            <h2 id={headingId} className="edu-module-title">
                <span className="edu-module-number" aria-hidden="true">{number}</span>
                <span className="edu-sr-only">{number} </span>
                <span>{title}</span>
            </h2>
            <p className="edu-module-status">
                <span className="edu-status-glyph" aria-hidden="true">{STATUS_ICON[status] ?? '○'}</span>
                <span className="edu-sr-only">{statusLabel}</span>
                <span aria-hidden="true">{statusShort}</span>
                <span className="edu-dot" aria-hidden="true">·</span>
                <span>{t('card.minutes', { minutes: module.minutes })}</span>
            </p>
            <p className="edu-module-goal">{t(`${module.id}.goal`)}</p>
            {module.datasets?.length > 0 && (
                <ul className="edu-dataset-chips" aria-label={t('card.datasets')}>
                    {module.datasets.map((id) => <li key={id}>{datasetName(id)}</li>)}
                </ul>
            )}
            {recommendedAfter && status === 'not-started' && <p className="edu-muted edu-module-recommended">{t('card.recommended', { number: recommendedAfter })}</p>}
            <div className="edu-module-actions">
                <button type="button" className={`df-button${status === 'in-progress' ? ' df-button-primary' : ''}`} onClick={() => onOpen(module)}
                    aria-label={t(`card.${action}Label`, { number, title })}>
                    {t(`card.${action}`)}<ChevronRight size={13} strokeWidth={2.5} aria-hidden="true" />
                </button>
            </div>
        </section>
    );
}
