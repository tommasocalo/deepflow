import { useId, useState } from 'react';
import { t } from '../lessons/engine/i18n.js';
import { softmaxWithTemperature, TEMPERATURE_SCORES } from './format.js';

/** Temperature divides the scores before softmax: below 1 sharpens, above 1 flattens. */
export default function SoftmaxTemperature() {
    const [temperature, setTemperature] = useState(1);
    const id = useId();
    const probabilities = softmaxWithTemperature(TEMPERATURE_SCORES.map((item) => item.score), temperature);
    return (
        <section className="edu-embed edu-temperature" aria-labelledby={`${id}-title`}>
            <h3 id={`${id}-title`} className="edu-section-label">{t('temperature.heading')}</h3>
            <label className="edu-temperature-control" htmlFor={`${id}-slider`}>
                <span>{t('temperature.label')}</span>
                <input id={`${id}-slider`} type="range" min="0.2" max="2" step="0.1" value={temperature}
                    onChange={(event) => setTemperature(Number(event.target.value))} aria-valuetext={temperature.toFixed(1)} />
                <output htmlFor={`${id}-slider`}>{temperature.toFixed(1)}</output>
            </label>
            <p className="edu-muted">{t('temperature.help')}</p>
            <table className="edu-table edu-temperature-table">
                <thead><tr><th scope="col">{t('temperature.character')}</th><th scope="col">{t('temperature.score')}</th><th scope="col">{t('temperature.probability')}</th></tr></thead>
                <tbody>
                    {TEMPERATURE_SCORES.map((item, index) => (
                        <tr key={item.token}>
                            <th scope="row"><code className="edu-code">{item.token}</code></th>
                            <td>{item.score.toFixed(1)}</td>
                            <td>
                                <span className="edu-bar" aria-hidden="true"><span style={{ width: `${(probabilities[index] * 100).toFixed(1)}%` }} /></span>
                                <span className="edu-bar-value">{(probabilities[index] * 100).toFixed(1)}%</span>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </section>
    );
}
