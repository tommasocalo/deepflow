import { tr } from '../text.js';

/**
 * The run's confusion matrix as a real table (docs/edu/SPEC.md §3.5): true classes in rows, predicted classes in
 * columns, counts as text and the cell shade by row fraction. Computed on the validation split (the training split
 * when there is none).
 */
export default function ConfusionMatrix({ record, compact = false, target = true }) {
    const matrix = record?.confusion;
    if (!matrix) {
        const classes = record?.numClasses;
        return <p className="edu-tr-empty">{classes && (classes < 2 || classes > 10) ? tr('confusion.notClasses') : tr('confusion.empty')}</p>;
    }
    if (matrix.length > 10) return <p className="edu-tr-empty">{tr('confusion.notClasses')}</p>;
    const names = record.classNames ?? matrix.map((_, index) => String(index));
    const total = matrix.flat().reduce((sum, value) => sum + value, 0);
    const caption = record.split?.val > 0 ? tr('confusion.caption', { count: total }) : tr('confusion.captionTrain', { count: total });
    return (
        <div className={`edu-tr-confusion${compact ? ' is-compact' : ''}`} data-edu-target={target ? 'trainer-confusion' : undefined}>
            <table>
                <caption>{caption}</caption>
                <thead>
                    <tr>
                        <th scope="col" className="edu-tr-corner">
                            <span className="edu-tr-sr">{tr('confusion.corner')}</span>
                            <span aria-hidden="true" className="edu-tr-corner-pred">{tr('confusion.predicted')} →</span>
                            <span aria-hidden="true" className="edu-tr-corner-true">{tr('confusion.true')} ↓</span>
                        </th>
                        {names.map((name) => <th key={name} scope="col" title={name}>{name}</th>)}
                    </tr>
                </thead>
                <tbody>
                    {matrix.map((row, truth) => {
                        const rowTotal = row.reduce((sum, value) => sum + value, 0) || 1;
                        return (
                            <tr key={names[truth]}>
                                <th scope="row" title={names[truth]}>{names[truth]}</th>
                                {row.map((count, predicted) => {
                                    const share = count / rowTotal;
                                    const dark = share > 0.5;
                                    return (
                                        <td key={names[predicted]} className={truth === predicted ? 'is-diagonal' : undefined}
                                            style={{ background: `rgba(17, 17, 17, ${(share * 0.88).toFixed(3)})`, color: dark ? '#fff' : '#111' }}>
                                            {count}
                                        </td>
                                    );
                                })}
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
