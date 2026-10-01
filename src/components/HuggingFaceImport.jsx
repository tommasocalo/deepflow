import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ArrowRight, ExternalLink, LoaderCircle, Search, X } from 'lucide-react';
import { parseModelReference, searchHubModels } from '../domain/huggingFace/client.js';
import { prepareHubModel, selectHubOnnxFile } from '../domain/huggingFace/index.js';
import './hugging-face-import.css';

const isAbort = (error) => error?.name === 'AbortError';
const formatCount = (value) => Number.isFinite(value) ? Intl.NumberFormat('en', { notation: 'compact' }).format(value) : null;

/** Insert-menu dialog. The parent owns the architecture-history transaction. */
export default function HuggingFaceImport({ open, onClose, onImport, currentArchitecture = null }) {
    const titleId = useId();
    const searchRef = useRef(null);
    const searchAbort = useRef(null);
    const previewAbort = useRef(null);
    const onCloseRef = useRef(onClose);
    onCloseRef.current = onClose;
    const [query, setQuery] = useState('gpt2');
    const [models, setModels] = useState([]);
    const [cursor, setCursor] = useState(null);
    const [searchBusy, setSearchBusy] = useState(false);
    const [searchError, setSearchError] = useState('');
    const [selectedId, setSelectedId] = useState('');
    const [preview, setPreview] = useState(null);
    const [sourceChoice, setSourceChoice] = useState('');
    const [previewBusy, setPreviewBusy] = useState(false);
    const [previewError, setPreviewError] = useState('');
    const [importBusy, setImportBusy] = useState(false);
    const [importError, setImportError] = useState('');

    useEffect(() => {
        if (!open) return undefined;
        const previousFocus = document.activeElement;
        const controller = new AbortController();
        searchAbort.current = controller;
        setSearchBusy(true);
        setQuery('gpt2');
        setModels([]);
        setCursor(null);
        setSelectedId('');
        setPreview(null);
        setSourceChoice('');
        setPreviewBusy(false);
        setImportBusy(false);
        setSearchError('');
        setPreviewError('');
        setImportError('');
        searchHubModels('gpt2', { signal: controller.signal })
            .then((result) => { if (!controller.signal.aborted) { setModels(result.models); setCursor(result.nextCursor); } })
            .catch((error) => { if (!isAbort(error)) setSearchError(error.message); })
            .finally(() => { if (!controller.signal.aborted) setSearchBusy(false); });
        requestAnimationFrame(() => searchRef.current?.focus());
        const onKeyDown = (event) => {
            if (event.key === 'Escape') { event.preventDefault(); onCloseRef.current(); }
        };
        document.addEventListener('keydown', onKeyDown);
        return () => {
            controller.abort();
            searchAbort.current?.abort();
            previewAbort.current?.abort();
            document.removeEventListener('keydown', onKeyDown);
            previousFocus?.focus?.();
        };
    }, [open]);

    if (!open) return null;

    const doSearch = async (event, nextCursor = null) => {
        event?.preventDefault();
        const text = query.trim();
        if (!text) return;
        if (!nextCursor) {
            const direct = parseModelReference(text);
            if (direct && (text.includes('/') || text.startsWith('https://'))) {
                openDirect(direct);
                return;
            }
        }
        searchAbort.current?.abort();
        const controller = new AbortController();
        searchAbort.current = controller;
        setSearchBusy(true);
        setSearchError('');
        if (!nextCursor) { previewAbort.current?.abort(); setPreviewBusy(false); setModels([]); setSelectedId(''); setPreview(null); }
        try {
            const result = await searchHubModels(text, { cursor: nextCursor, signal: controller.signal });
            if (!controller.signal.aborted) {
                setModels((previous) => nextCursor ? [...previous, ...result.models] : result.models);
                setCursor(result.nextCursor);
            }
        } catch (error) {
            if (!isAbort(error)) setSearchError(error.message);
        } finally {
            if (!controller.signal.aborted) setSearchBusy(false);
        }
    };

    const showPreview = async (reference) => {
        previewAbort.current?.abort();
        const controller = new AbortController();
        previewAbort.current = controller;
        const modelId = typeof reference === 'string' ? reference : reference.repoId;
        setSelectedId(modelId);
        setPreview(null);
        setSourceChoice('');
        setPreviewBusy(true);
        setPreviewError('');
        setImportError('');
        try {
            const result = await prepareHubModel(reference, { signal: controller.signal });
            if (!controller.signal.aborted) { setPreview(result); setSourceChoice(result.onnx?.filePath ?? ''); }
        } catch (error) {
            if (!isAbort(error)) setPreviewError(error.message);
        } finally {
            if (!controller.signal.aborted) setPreviewBusy(false);
        }
    };

    const chooseSourceGraph = async (filePath) => {
        if (!filePath || !preview) return;
        previewAbort.current?.abort();
        const controller = new AbortController();
        previewAbort.current = controller;
        setPreviewBusy(true);
        setSourceChoice(filePath);
        setPreviewError('');
        setImportError('');
        // A pending choice must never import the previously inspected graph.
        setPreview((previous) => ({ ...previous, onnx: null, requiresOnnxSelection: false,
            inspection: { family: previous.inspection.family, supported: false } }));
        try {
            const result = await selectHubOnnxFile(preview, filePath, { signal: controller.signal });
            if (!controller.signal.aborted) setPreview(result);
        } catch (error) {
            if (!isAbort(error)) setPreviewError(error.message);
        } finally {
            if (!controller.signal.aborted) setPreviewBusy(false);
        }
    };

    const openDirect = (reference, nextQuery = null) => {
        searchAbort.current?.abort();
        setSearchBusy(false);
        setModels([{ id: reference.repoId }]);
        setCursor(null);
        setSearchError('');
        if (nextQuery) setQuery(nextQuery);
        void showPreview(reference);
    };

    const confirmImport = async () => {
        if (!preview?.inspection.supported || importBusy || previewBusy) return;
        setImportBusy(true);
        setImportError('');
        try { await onImport(preview); onClose(); }
        catch (error) { setImportError(error?.message || 'Architecture import failed.'); }
        finally { setImportBusy(false); }
    };

    const inspection = preview?.inspection;
    const importedDimensions = inspection?.supported ? inspection.configuration : null;
    const dimensions = importedDimensions && currentArchitecture ? [
        { label: 'blocks', current: currentArchitecture.nLayers, imported: importedDimensions.nLayers },
        { label: 'width', current: currentArchitecture.dModel, imported: importedDimensions.dModel },
        { label: 'heads', current: currentArchitecture.nHeads, imported: importedDimensions.nHeads },
    ].filter((item) => Number.isFinite(item.current)) : [];
    const changedDimensions = dimensions.filter((item) => item.current !== item.imported);
    const sourceGraph = preview?.onnx?.operatorGraph;
    const sourceIssues = preview?.onnx?.recognition?.issues ?? [];
    return createPortal(
        <div className="df-hf-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
            <section className="df-hf-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
                <header className="df-hf-header">
                    <div><span className="df-hf-header-kicker">MODEL ARCHITECTURE</span><strong id={titleId}>Import from Hugging Face</strong><span>Find a model, preview its structure, then open it in DeepFlow.</span></div>
                    <button className="df-hf-close" type="button" aria-label="Close Hugging Face import" onClick={onClose}><X size={18} /></button>
                </header>

                <form className="df-hf-search" onSubmit={doSearch}>
                    <div className="df-hf-search-field">
                        <Search size={18} aria-hidden="true" />
                        <input ref={searchRef} type="search" aria-label="Search Hugging Face models or paste a model URL"
                            placeholder="Model name, ID or Hugging Face URL" value={query} onChange={(event) => setQuery(event.target.value)} />
                        {query && <button className="df-hf-search-clear" type="button" aria-label="Clear model search" title="Clear search"
                            onClick={() => { setQuery(''); searchRef.current?.focus(); }}><X size={14} /></button>}
                    </div>
                    <button className="df-hf-search-submit" type="submit">Search <ArrowRight size={14} aria-hidden="true" /></button>
                </form>
                <div className="df-hf-search-example"><span>Quick models</span>
                    <button type="button" aria-pressed={selectedId === 'openai-community/gpt2'}
                        onClick={() => openDirect({ repoId: 'openai-community/gpt2' }, 'openai-community/gpt2')}>GPT-2 decoder</button>
                    <button type="button" aria-pressed={selectedId === 'google-bert/bert-base-uncased'}
                        onClick={() => openDirect({ repoId: 'google-bert/bert-base-uncased' }, 'google-bert/bert-base-uncased')}>
                        BERT masked LM <ArrowRight size={12} aria-hidden="true" />
                    </button>
                    <button type="button" aria-pressed={selectedId === 'vtemplier/LeNet_MNIST'}
                        onClick={() => openDirect({ repoId: 'vtemplier/LeNet_MNIST' }, 'vtemplier/LeNet_MNIST')}>LeNet vision</button>
                </div>

                <div className="df-hf-body">
                    <div className="df-hf-results" aria-label="Hugging Face model results">
                        <div className="df-hf-section-title">Models {models.length ? <span>{models.length}</span> : null}</div>
                        {searchError && <p className="df-hf-error" role="alert">{searchError}</p>}
                        {!searchBusy && !models.length && !searchError && <p className="df-hf-empty">No models found. Try a model ID or URL.</p>}
                        {models.map((model) => <button key={model.id} type="button" className="df-hf-result" aria-selected={selectedId === model.id}
                            onClick={() => showPreview(model.id)}>
                            <strong>{model.id}</strong>
                            <span>{[model.pipelineTag, formatCount(model.downloads) && `${formatCount(model.downloads)} downloads`].filter(Boolean).join(' · ') || 'Preview architecture'}</span>
                        </button>)}
                        {searchBusy && <p className="df-hf-loading"><LoaderCircle size={14} className="df-hf-spinner" /> Loading models…</p>}
                        {cursor && !searchBusy && <button type="button" className="df-hf-more" onClick={() => doSearch(null, cursor)}>Load more models <ArrowRight size={13} /></button>}
                    </div>

                    <div className="df-hf-preview" aria-live="polite">
                        {!selectedId && <div className="df-hf-placeholder"><span>HF</span><strong>Select a model</strong><p>DeepFlow checks its configuration and available exported graphs.</p></div>}
                        {previewBusy && <p className="df-hf-loading" role="status"><LoaderCircle size={16} className="df-hf-spinner" /> Reading model architecture…</p>}
                        {previewError && <p className="df-hf-error" role="alert">{previewError}</p>}
                        {preview && <>
                            <div className="df-hf-preview-head">
                                <div><span className="df-hf-eyebrow">{sourceGraph && !preview.config ? 'Published architecture' : inspection.family}</span><h3>{preview.repoId}</h3><span className="df-hf-commit">{preview.requestedRevision} · {preview.commit.slice(0, 12)}</span></div>
                                <span className={`df-hf-status ${inspection.supported ? 'is-supported' : preview.requiresOnnxSelection || previewBusy ? 'is-pending' : ''}`}>{previewBusy ? 'Checking…' : preview.requiresOnnxSelection ? 'Choose graph' : inspection.supported ? 'Supported' : 'Not mapped yet'}</span>
                            </div>
                            {preview.onnxFiles?.length > 1 && !inspection.configuration && <div className="df-hf-source-picker">
                                <label htmlFor={`${titleId}-graph`}>Source graph</label>
                                <select id={`${titleId}-graph`} aria-label="Hugging Face source graph" value={sourceChoice}
                                    onChange={(event) => void chooseSourceGraph(event.target.value)} disabled={importBusy}>
                                    <option value="" disabled>Choose a graph…</option>
                                    {preview.onnxFiles.map((file) => <option key={file.path} value={file.path}>{file.path}</option>)}
                                </select>
                                <p>This repository has multiple exports or model parts. Import uses the graph you choose.</p>
                            </div>}
                            {!inspection.supported && !previewBusy && inspection.reason && <p className={preview.requiresOnnxSelection ? 'df-hf-source-note' : 'df-hf-unsupported'}>{inspection.reason}</p>}
                            {sourceGraph && <div className="df-hf-facts" aria-label="Source graph summary">
                                <div><span>Operations</span><strong>{sourceGraph.nodes.length}</strong></div>
                                <div><span>Components</span><strong>{preview.onnx.recognition.components.length}</strong></div>
                                <div><span>Inputs</span><strong>{sourceGraph.inputs.filter((item) => !item.defaultInitializer).length}</strong></div>
                                <div><span>Outputs</span><strong>{sourceGraph.outputs.length}</strong></div>
                            </div>}
                            {inspection.supported && <>
                                {inspection.configuration && <div className="df-hf-facts">
                                    <div><span>Width</span><strong>{inspection.configuration.dModel}</strong></div>
                                    <div><span>Blocks</span><strong>{inspection.configuration.nLayers}</strong></div>
                                    <div><span>Heads</span><strong>{inspection.configuration.nHeads}</strong></div>
                                    <div><span>Context</span><strong>{inspection.configuration.context}</strong></div>
                                </div>}
                                {dimensions.length > 0 && <div className="df-hf-impact" aria-label="Comparison with current architecture">
                                    <strong>Compared with current graph</strong>
                                    {changedDimensions.length ? <p>{changedDimensions.map((item) => `${item.label} ${item.current} → ${item.imported}`).join(' · ')}</p>
                                        : <p>Dimensions match. Import replaces the graph with {inspection.family} components and records their source; an identical layout may look unchanged.</p>}
                                </div>}
                                <div className="df-hf-section-title">Editable components</div>
                                <ul className={`df-hf-components${sourceGraph ? ' is-operator-mapping' : ''}`}>{inspection.components.map((component, index) => <li key={`${component.name}-${index}`}><strong>{component.name}</strong><span>{component.detail}</span></li>)}</ul>
                                <div className="df-hf-limits"><strong>Import scope</strong>{inspection.limitations.filter((item) => !item.startsWith('Operator graph only:')).map((item) => <p key={item}>{item}</p>)}</div>
                            </>}
                            {sourceGraph && <details className="df-hf-config df-hf-source-map">
                                <summary>Inspect source mapping <span>{preview.onnx.filePath}</span></summary>
                                <div className="df-hf-source-map-body">
                                    <p>ONNX operations are mapped to editable DeepFlow components. The original graph stays attached after import.</p>
                                    <ul>{preview.onnx.recognition.components.map((component, index) => <li key={index}>
                                        <strong>{component.kind}</strong><span>{component.source.nodeNames.join(' + ')}</span>
                                    </li>)}</ul>
                                    {sourceIssues.length > 0 && <><strong>Unmapped operations</strong><ul>{sourceIssues.slice(0, 30).map((issue, index) => <li key={index}>{issue.message}</li>)}</ul>
                                        {sourceIssues.length > 30 && <p>{sourceIssues.length - 30} more mapping issues.</p>}</>}
                                </div>
                            </details>}
                            {preview.config && <details className="df-hf-config">
                                <summary>Inspect config.json <span>{Object.keys(preview.config).length} fields</span></summary>
                                <pre>{JSON.stringify(preview.config, null, 2)}</pre>
                            </details>}
                            <a className="df-hf-external" href={preview.url} target="_blank" rel="noopener noreferrer">View source revision <ExternalLink size={12} /></a>
                        </>}
                    </div>
                </div>

                {importError && <div className="df-hf-import-error" role="alert">
                    <div><strong>Import failed</strong><span>{importError} Your current architecture is preserved.</span></div>
                    <button type="button" onClick={confirmImport}>Try again</button>
                </div>}
                <footer className="df-hf-footer">
                    <span>Import replaces the current graph and clears training. Pretrained weights are not loaded.</span>
                    <button type="button" className="df-hf-import" onClick={confirmImport} disabled={!inspection?.supported || importBusy || previewBusy}>
                        {importBusy ? 'Importing…' : 'Import architecture'} <ArrowRight size={14} />
                    </button>
                </footer>
            </section>
        </div>,
        document.body,
    );
}
