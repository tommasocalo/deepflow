import { useEffect, useRef } from 'react';
import { ChevronRight } from 'lucide-react';
import { t } from '../lessons/engine/i18n.js';
import RichText from './RichText.jsx';

/** Three takeaways, the next module and the way back to Learn (docs/edu/SPEC.md §3.3). */
export default function RecapView({ module, next, facts, complete, onNext, onLearn }) {
    const headingRef = useRef(null);
    useEffect(() => { headingRef.current?.focus(); }, [module.id]);
    return (
        <section className="edu-recap" aria-labelledby="edu-recap-heading">
            <p className="edu-subgoal">{t('recap.heading')}</p>
            <h2 id="edu-recap-heading" ref={headingRef} tabIndex={-1} className="edu-step-title">{t('recap.intro', { title: t(`${module.id}.title`) })}</h2>
            <ol className="edu-takeaways">
                {[1, 2, 3].map((index) => <li key={index}><RichText text={t(`${module.id}.recap.${index}`)} params={facts} /></li>)}
            </ol>
            {!complete && <p className="edu-muted">{t('recap.unfinished')}</p>}
            <footer className="edu-nav edu-nav-stack">
                {next && (
                    <button type="button" className="df-button df-button-primary edu-hard" onClick={onNext}>
                        {t('recap.next', { title: t(`${next.id}.title`) })}<ChevronRight size={13} strokeWidth={2.5} aria-hidden="true" />
                    </button>
                )}
                <button type="button" className="df-button" onClick={onLearn}>{t('recap.learn')}</button>
            </footer>
        </section>
    );
}
