import { useId, useState } from 'react';
import { focusNode } from '../../lessons/engine/canvasBridge.js';
import { useEduUiStore } from '../../shell/uiStore.js';
import { getDataset } from '../../data/registry.js';
import { useEduDataStore } from '../../data/store.js';
import { OPTIMIZER_TYPES } from '../optimizers.js';
import { formatLr } from '../charts.js';
import { groupIssues, runStatusText } from '../describe.js';
import { SETTING_LIMITS, useTrainerStore } from '../store.js';
import { tr } from '../text.js';
import { patchOptimizer, useObjective } from './hooks.js';

const DEFAULT_LR = 0.01;

/**
 * A number input that keeps what the learner is typing ("0.0", "") and commits only valid values, so a controlled
 * value never fights the keyboard. `parse` returns a number or null.
 */
function DraftNumber({ value, onCommit, parse, className, ...props }) {
    const [draft, setDraft] = useState(null);
    return (
        <input
            {...props}
            type="number"
            className={className}
            value={draft ?? String(value)}
            onChange={(event) => {
                setDraft(event.target.value);
                const parsed = parse(event.target.value);
                if (parsed !== null) onCommit(parsed);
            }}
            onBlur={() => setDraft(null)}
        />
    );
}

const positive = (text) => {
    const value = Number(text);
    return text.trim() !== '' && Number.isFinite(value) && value > 0 ? value : null;
};
const wholeIn = ({ min, max }) => (text) => {
    const value = Number(text);
    return text.trim() !== '' && Number.isInteger(value) && value >= min && value <= max ? value : null;
};

/** The learning rate of the training graph's optimizer (data-edu-target="trainer-lr"). */
export function LrField({ compact = false, disabled = false }) {
    const { optimizer } = useObjective();
    const id = useId();
    const lr = Number(optimizer?.hyperparameters?.learning_rate ?? DEFAULT_LR);
    return (
        <span className={`edu-tr-field${compact ? ' is-compact' : ''}`}>
            <label htmlFor={id}>{compact ? tr('run.lr') : tr('run.lrLabel')}</label>
            <DraftNumber
                id={id}
                className="df-input edu-tr-input edu-tr-lr"
                data-edu-target="trainer-lr"
                min="0"
                step="any"
                disabled={disabled}
                value={formatLr(lr)}
                parse={positive}
                onCommit={(value) => patchOptimizer({ learningRate: value })}
            />
        </span>
    );
}

export function OptimizerField() {
    const { optimizer } = useObjective();
    const id = useId();
    const type = optimizer?.type ?? 'Adam';
    return (
        <span className="edu-tr-field">
            <label htmlFor={id}>{tr('run.optimizer')}</label>
            <select id={id} className="df-input edu-tr-input edu-tr-select" value={type} onChange={(event) => patchOptimizer({ type: event.target.value })}>
                {OPTIMIZER_TYPES.map((value) => (
                    <option key={value} value={value}>{optimizer ? value : tr('run.optimizerDefault', { type: value })}</option>
                ))}
            </select>
        </span>
    );
}

function SettingField({ name, label }) {
    const value = useTrainerStore((state) => state.settings[name]);
    const id = useId();
    const limits = SETTING_LIMITS[name];
    return (
        <span className="edu-tr-field">
            <label htmlFor={id} title={name === 'seed' ? tr('run.seedHelp') : undefined}>{label}</label>
            <DraftNumber
                id={id}
                className="df-input edu-tr-input"
                min={limits.min}
                max={limits.max}
                step="1"
                value={value}
                parse={wholeIn(limits)}
                onCommit={(next) => useTrainerStore.getState().setSettings({ [name]: next })}
            />
        </span>
    );
}

/** "Show on canvas": open the Architecture tab and focus the block (canvas bridge, docs/edu/SPEC.md §6.12). */
function showOnCanvas(nodeId) {
    useEduUiStore.getState().setTab('architecture');
    setTimeout(() => focusNode(nodeId), 80);
}

/** Problems the learner fixes in the model or data; "trains in Colab" cases are guidance, not mistakes. */
const COLAB_CODES = new Set(['dataset-export-only', 'unsupported-op', 'too-large', 'multi-phase', 'custom-loss']);

function IssueList({ issues, limit, colab = false }) {
    return (
        <ul>
            {issues.slice(0, limit).map((issue) => (
                <li key={`${issue.code}:${issue.message}`}>
                    {issue.message}
                    {issue.nodeId && (
                        <> <button type="button" className="edu-tr-link" onClick={() => showOnCanvas(issue.nodeId)}>{tr('preflight.showOnCanvas')}</button></>
                    )}
                    {colab && (
                        <> <button type="button" className="edu-tr-link" onClick={() => useEduUiStore.getState().openCode('train')}>{tr('preflight.openTrainPy')}</button></>
                    )}
                </li>
            ))}
        </ul>
    );
}

export function PreflightList({ preflight, limit = 4 }) {
    const all = groupIssues(preflight?.errors ?? []);
    const colab = all.filter((issue) => COLAB_CODES.has(issue.code));
    const errors = all.filter((issue) => !COLAB_CODES.has(issue.code));
    const warnings = preflight?.warnings ?? [];
    if (!all.length && !warnings.length) return null;
    return (
        <div className="edu-tr-preflight">
            {errors.length > 0 && (
                <div className="edu-tr-issues is-error">
                    <p className="edu-tr-issues-title">{tr('preflight.errors')}</p>
                    <IssueList issues={errors} limit={limit} />
                </div>
            )}
            {colab.length > 0 && (
                <div className="edu-tr-issues is-colab">
                    <p className="edu-tr-issues-title">{tr('preflight.colab')}</p>
                    <IssueList issues={colab} limit={limit} colab />
                </div>
            )}
            {warnings.length > 0 && (
                <div className="edu-tr-issues is-warning">
                    <p className="edu-tr-issues-title">{tr('preflight.warnings')}</p>
                    <IssueList issues={warnings} limit={limit} />
                </div>
            )}
        </div>
    );
}

/** Train / Stop / Check initial loss. `locked` keeps Train closed behind a lesson's prediction gate. */
export function RunButtons({ locked = false, lockedReason = '', compact = false }) {
    const status = useTrainerStore((state) => state.status);
    const running = status === 'running' || status === 'stopping';
    const store = useTrainerStore.getState;
    return (
        <>
            <div className={`edu-tr-buttons${compact ? ' is-compact' : ''}`}>
                <button type="button" className="df-button df-button-primary edu-tr-train" data-edu-target="trainer-run"
                    disabled={locked || running} onClick={() => store().start()}>
                    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M2 1l7 4-7 4z" fill="currentColor" /></svg>
                    {tr('run.train')}
                </button>
                <button type="button" className="df-button edu-tr-stop" disabled={!running || status === 'stopping'} onClick={() => store().stop()}>
                    <svg width="9" height="9" viewBox="0 0 9 9" aria-hidden="true"><rect x="1" y="1" width="7" height="7" fill="currentColor" /></svg>
                    {tr('run.stop')}
                </button>
                <button type="button" className="df-button edu-tr-init" data-edu-target="trainer-init-loss" disabled={locked || running}
                    onClick={() => store().evaluateInit()}>
                    {tr('run.initLoss')}
                </button>
            </div>
            {locked && <p className="edu-tr-locked">{lockedReason || tr('run.locked')}</p>}
        </>
    );
}

/** The left column of the TrainerPanel (docs/edu/SPEC.md §3.5). */
export default function RunSettings({ locked = false, lockedReason = '' }) {
    const preflight = useTrainerStore((state) => state.preflight);
    const status = useTrainerStore((state) => state.status);
    const current = useTrainerStore((state) => state.current);
    const dataName = useEduDataStore((state) => getDataset(state.datasetId)?.name ?? null);
    const seed = useEduDataStore((state) => state.current().dataSeed);
    let model = tr('run.modelUnknown');
    const problems = groupIssues(preflight.errors).length;
    if (preflight.at && problems) model = problems === 1 ? tr('run.modelError1') : tr('run.modelErrors', { count: problems });
    else if (preflight.at && Number.isFinite(preflight.paramCount)) model = tr('run.modelOk', { params: preflight.paramCount.toLocaleString('en-US') });
    return (
        <div className="edu-tr-settings">
            <dl className="edu-tr-facts">
                <div>
                    <dt>{tr('run.dataset')}</dt>
                    <dd>
                        {dataName ? tr('run.datasetValue', { name: dataName, seed }) : (
                            <>
                                {tr('run.noDataset')}{' '}
                                <button type="button" className="edu-tr-link" onClick={() => useEduUiStore.getState().setTab('data')}>{tr('run.chooseDataset')}</button>
                            </>
                        )}
                    </dd>
                </div>
                <div>
                    <dt>{tr('run.model')}</dt>
                    <dd>{model}</dd>
                </div>
            </dl>
            <div className="edu-tr-row">
                <OptimizerField />
                <LrField />
            </div>
            <div className="edu-tr-row is-three">
                <SettingField name="epochs" label={tr('run.epochs')} />
                <SettingField name="batchSize" label={tr('run.batch')} />
                <SettingField name="seed" label={tr('run.seed')} />
            </div>
            <RunButtons locked={locked} lockedReason={lockedReason} />
            <p className="edu-tr-status">{runStatusText(current, status)}</p>
            {/* Behind an open prediction gate the issue text could give the answer away; it shows once answered. */}
            {!locked && <PreflightList preflight={preflight} />}
        </div>
    );
}
