import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, ArrowRight, Check, FileUp, LoaderCircle, X } from 'lucide-react';
import { recognizeOnnxComponents } from '../domain/onnx/editable.js';
import './onnx-import.css';

const MAX_FILE_BYTES = 64 * 1024 * 1024;

function countLabel(count, singular, plural = `${singular}s`) {
    return `${count} ${count === 1 ? singular : plural}`;
}

/** Inspect an operator graph before replacing the current architecture. */
export default function OnnxImport({ open, onClose, onImport, currentSource = null }) {
    const titleId = useId();
    const inputRef = useRef(null);
    const browseRef = useRef(null);
    const loadId = useRef(0);
    const onCloseRef = useRef(onClose);
    onCloseRef.current = onClose;
    const [fileName, setFileName] = useState('');
    const [operatorGraph, setOperatorGraph] = useState(null);
    const [recognition, setRecognition] = useState(null);
    const [readBusy, setReadBusy] = useState(false);
    const [importBusy, setImportBusy] = useState(false);
    const [readError, setReadError] = useState('');
    const [importError, setImportError] = useState('');
    const [dragging, setDragging] = useState(false);
    const [viewingSavedSource, setViewingSavedSource] = useState(false);
    const [visibleOps, setVisibleOps] = useState(60);
    const [visibleIssues, setVisibleIssues] = useState(20);

    useEffect(() => {
        if (!open) return undefined;
        const previousFocus = document.activeElement;
        loadId.current += 1;
        setFileName('');
        setOperatorGraph(null);
        setRecognition(null);
        setReadBusy(false);
        setImportBusy(false);
        setReadError('');
        setImportError('');
        setDragging(false);
        setViewingSavedSource(false);
        setVisibleOps(60);
        setVisibleIssues(20);
        requestAnimationFrame(() => browseRef.current?.focus());
        const onKeyDown = (event) => {
            if (event.key === 'Escape') { event.preventDefault(); onCloseRef.current(); }
        };
        document.addEventListener('keydown', onKeyDown);
        return () => {
            loadId.current += 1;
            document.removeEventListener('keydown', onKeyDown);
            previousFocus?.focus?.();
        };
    }, [open]);

    if (!open) return null;

    const loadFile = async (file) => {
        if (!file) return;
        const request = ++loadId.current;
        setFileName(file.name);
        setOperatorGraph(null);
        setRecognition(null);
        setReadError('');
        setImportError('');
        setReadBusy(true);
        setViewingSavedSource(false);
        setVisibleOps(60);
        setVisibleIssues(20);
        try {
            if (!file.size || file.size > MAX_FILE_BYTES) {
                throw new Error('Choose a non-empty .onnx graph file up to 64 MiB. External weight data may be referenced but is not loaded.');
            }
            const bytes = await file.arrayBuffer();
            if (request !== loadId.current) return;
            // The protobuf parser and schema are only needed after a file is
            // selected; keep the ordinary architecture editor quick to load.
            const { readOnnxModel } = await import('../domain/onnx/index.js');
            if (request !== loadId.current) return;
            const graph = readOnnxModel(bytes);
            const mapped = recognizeOnnxComponents(graph);
            if (request !== loadId.current) return;
            setOperatorGraph(graph);
            setRecognition(mapped);
        } catch (error) {
            if (request === loadId.current) setReadError(error?.message || 'The ONNX file could not be read.');
        } finally {
            if (request === loadId.current) setReadBusy(false);
        }
    };

    const confirmImport = async () => {
        if (!operatorGraph || !recognition?.supported || importBusy || viewingSavedSource) return;
        setImportBusy(true);
        setImportError('');
        try { await onImport(operatorGraph, fileName); onClose(); }
        catch (error) { setImportError(error?.message || 'ONNX import failed.'); }
        finally { setImportBusy(false); }
    };

    const inspectSavedSource = () => {
        if (!currentSource?.operatorGraph) return;
        const savedGraph = { ...currentSource.operatorGraph, format: 'onnx-operator-graph' };
        setFileName(currentSource.fileName || `${savedGraph.name}.onnx`);
        setOperatorGraph(savedGraph);
        setRecognition(recognizeOnnxComponents(savedGraph));
        setReadError('');
        setImportError('');
        setViewingSavedSource(true);
        setVisibleOps(60);
        setVisibleIssues(20);
    };

    const opset = operatorGraph?.opsets.find((item) => !item.domain);
    return createPortal(
        <div className="df-onnx-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
            <section className="df-onnx-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
                <header className="df-onnx-header">
                    <div><span className="df-onnx-kicker">OPERATOR GRAPH</span><h2 id={titleId}>{viewingSavedSource ? 'ONNX source' : 'Import ONNX'}</h2>
                        <p>{viewingSavedSource ? 'Original operators and the components recognized at import.'
                            : 'Inspect the real operators and their editable DeepFlow mapping before importing.'}</p></div>
                    <button type="button" className="df-onnx-close" aria-label="Close ONNX import" onClick={onClose}><X size={17} /></button>
                </header>

                <div className="df-onnx-content">
                    <div className={`df-onnx-upload${dragging ? ' is-dragging' : ''}`}
                        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                        onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }}
                        onDrop={(event) => { event.preventDefault(); setDragging(false); void loadFile(event.dataTransfer.files?.[0]); }}>
                        <span className="df-onnx-file-icon"><FileUp size={19} strokeWidth={1.8} /></span>
                        <div className="df-onnx-file-text"><strong>{fileName || 'Choose an ONNX model'}</strong>
                            <span>{fileName ? 'Choose another file or drop it here' : 'Local .onnx graph · up to 64 MiB · or drop it here'}</span></div>
                        <button ref={browseRef} type="button" className="df-onnx-file-button"
                            onClick={() => inputRef.current?.click()}>{fileName ? 'Replace file' : 'Browse file'}</button>
                        <input ref={inputRef} id={`${titleId}-file`} type="file" tabIndex={-1} accept=".onnx,application/octet-stream"
                            aria-label="Choose an ONNX file" onChange={(event) => { void loadFile(event.target.files?.[0]); event.target.value = ''; }} />
                    </div>
                    {readBusy && <p className="df-onnx-reading" role="status"><LoaderCircle size={14} className="df-onnx-spinner" /> Reading operator graph…</p>}
                    {readError && <p className="df-onnx-error" role="alert"><AlertTriangle size={14} />{readError}</p>}

                    {!operatorGraph && !readBusy && !readError && <div className="df-onnx-empty">
                        <strong>Two views of the same model</strong>
                        <p>The file stays on this device. DeepFlow will show the ONNX operators and which ones it can turn into editable components.</p>
                        {currentSource?.operatorGraph && <button type="button" onClick={inspectSavedSource}>Inspect imported ONNX source <ArrowRight size={12} /></button>}
                    </div>}

                    {operatorGraph && recognition && <>
                        <div className="df-onnx-model-head">
                            <div><span className="df-onnx-small-label">{viewingSavedSource ? 'Original imported source · later edits are not reflected' : fileName}</span><h3>{operatorGraph.name}</h3></div>
                            <span className={`df-onnx-status${viewingSavedSource ? ' is-snapshot' : recognition.supported ? ' is-ready' : ''}`}>
                                {viewingSavedSource || recognition.supported ? <Check size={12} /> : <AlertTriangle size={12} />}
                                {viewingSavedSource ? 'Imported source snapshot' : recognition.supported ? 'Editable mapping complete' : 'Operator view only'}
                            </span>
                        </div>
                        <div className="df-onnx-facts">
                            <div><span>Operators</span><strong>{operatorGraph.nodes.length}</strong></div>
                            <div><span>Inputs / outputs</span><strong>{operatorGraph.inputs.length} / {operatorGraph.outputs.length}</strong></div>
                            <div><span>Initializers</span><strong>{operatorGraph.initializers.length}</strong></div>
                            <div><span>ONNX opset</span><strong>{opset?.version ?? '—'}</strong></div>
                        </div>
                        <div className="df-onnx-mapping" aria-label="ONNX operator to DeepFlow component mapping">
                            <section className="df-onnx-column" aria-label="ONNX operators">
                                <header><span className="df-onnx-column-tag">01</span><div><strong>ONNX operators</strong><span>From the file · {countLabel(operatorGraph.nodes.length, 'operation')}</span></div></header>
                                <ol className="df-onnx-operator-list">{operatorGraph.nodes.slice(0, visibleOps).map((node) => <li key={node.id}>
                                    <span className="df-onnx-op-index">{String(node.index + 1).padStart(2, '0')}</span>
                                    <div><strong>{node.domain ? `${node.domain}::` : ''}{node.opType}</strong><span>{node.name}</span>
                                        <small>{node.inputs.filter(Boolean).join(', ') || '∅'} → {node.outputs.filter(Boolean).join(', ') || '∅'}</small></div>
                                </li>)}{operatorGraph.nodes.length > visibleOps && <li className="df-onnx-more-row"><button type="button" onClick={() => setVisibleOps((count) => count + 60)}>
                                    Show next {Math.min(60, operatorGraph.nodes.length - visibleOps)} of {operatorGraph.nodes.length} operators
                                </button></li>}</ol>
                            </section>
                            <section className="df-onnx-column" aria-label="DeepFlow editable components">
                                <header><span className="df-onnx-column-tag is-yellow"><ArrowRight size={14} /></span><div><strong>DeepFlow components</strong><span>{countLabel(recognition.components.length, 'recognized block')}</span></div></header>
                                {recognition.components.length > 0 ? <ol className="df-onnx-component-list">{recognition.components.map((component, index) => <li key={`${component.label}-${index}`}>
                                    <span className="df-onnx-component-mark">{String(index + 1).padStart(2, '0')}</span>
                                    <div><strong>{component.kind}</strong><span>{component.label}</span>
                                        <small>From {component.source.nodeNames.join(' + ')}</small></div>
                                </li>)}</ol> : <p className="df-onnx-no-components">No editable component was recognized in this graph.</p>}
                                {recognition.issues.length > 0 && <div className="df-onnx-issues" role="status">
                                    <strong>{countLabel(recognition.issues.length, 'mapping issue')}</strong>
                                    <ul>{recognition.issues.slice(0, visibleIssues).map((issue, index) => <li key={`${issue.code}-${issue.nodeId}-${index}`}>
                                        {issue.message}
                                    </li>)}{recognition.issues.length > visibleIssues && <li><button type="button" onClick={() => setVisibleIssues((count) => count + 20)}>
                                        Show next {Math.min(20, recognition.issues.length - visibleIssues)} issues
                                    </button></li>}</ul>
                                </div>}
                            </section>
                        </div>
                        <div className="df-onnx-scope">
                            <strong>Import scope</strong>
                            <span>Editable structure only; pretrained weights are not loaded. The operator graph and source mapping stay attached for inspection.</span>
                            {operatorGraph.limitations.filter((item) => !item.startsWith('Operator graph only:')).map((item) => <span key={item}>{item}</span>)}
                        </div>
                    </>}
                </div>

                {importError && <p className="df-onnx-import-error" role="alert"><AlertTriangle size={14} />{importError} The current architecture was preserved.</p>}
                <footer className="df-onnx-footer">
                    <span>{viewingSavedSource ? 'Original imported graph · choose a file to make a new import.' : operatorGraph ? recognition?.supported ? `All ${operatorGraph.nodes.length} operators mapped.`
                        : 'Import is unavailable until every operator has an editable mapping.' : 'Choose a local ONNX file to preview its mapping.'}</span>
                    <button type="button" className="df-onnx-import" disabled={!recognition?.supported || readBusy || importBusy || viewingSavedSource} onClick={confirmImport}>
                        {importBusy ? 'Importing…' : 'Import editable graph'} <ArrowRight size={14} />
                    </button>
                </footer>
            </section>
        </div>, document.body,
    );
}
