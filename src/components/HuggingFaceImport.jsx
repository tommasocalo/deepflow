import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ArrowRight, ExternalLink, LoaderCircle, Search, X } from 'lucide-react';
import { fetchHubModelConfig, parseModelReference, searchHubModels } from '../domain/huggingFace/client.js';
import { inspectHubModel } from '../domain/huggingFace/index.js';
import './hugging-face-import.css';

const isAbort = (error) => error?.name === 'AbortError';
const formatCount = (value) => Number.isFinite(value) ? Intl.NumberFormat('en', { notation: 'compact' }).format(value) : null;

/** Insert-menu dialog. The parent owns the architecture-history transaction. */
export default function HuggingFaceImport({ open, onClose, onImport }) {
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
        setModels([]);
        setCursor(null);
        setSelectedId('');
        setPreview(null);
        setSearchError('');
        setPreviewError('');
        setImportError('');
        searchHubModels('gpt2', { signal: controller.signal })
            .then((result) => { setModels(result.models); setCursor(result.nextCursor); })
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
                setModels([{ id: direct.repoId }]);
                setCursor(null);
                setSearchError('');
                void showPreview(direct);
                return;
            }
        }
        searchAbort.current?.abort();
        const controller = new AbortController();
        searchAbort.current = controller;
        setSearchBusy(true);
        setSearchError('');
        if (!nextCursor) { setModels([]); setSelectedId(''); setPreview(null); }
        try {
            const result = await searchHubModels(text, { cursor: nextCursor, signal: controller.signal });
            setModels((previous) => nextCursor ? [...previous, ...result.models] : result.models);
            setCursor(result.nextCursor);
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
        setPreviewBusy(true);
        setPreviewError('');
        setImportError('');
        try {
            const result = await fetchHubModelConfig(reference, { signal: controller.signal });
            if (!controller.signal.aborted) setPreview(inspectHubModel(result));
        } catch (error) {
            if (!isAbort(error)) setPreviewError(error.message);
        } finally {
            if (!controller.signal.aborted) setPreviewBusy(false);
        }
    };

    const confirmImport = async () => {
        if (!preview?.inspection.supported || importBusy) return;
        setImportBusy(true);
        setImportError('');
        try { await onImport(preview); onClose(); }
        catch (error) { setImportError(error?.message || 'Architecture import failed.'); }
        finally { setImportBusy(false); }
    };

    const inspection = preview?.inspection;
    return createPortal(
        <div className="df-hf-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
            <section className="df-hf-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
                <header className="df-hf-header">
                    <div><strong id={titleId}>Import from Hugging Face</strong><span>Explore a model configuration, then open it as editable architecture.</span></div>
                    <button className="df-hf-close" type="button" aria-label="Close Hugging Face import" onClick={onClose}><X size={18} /></button>
                </header>

                <form className="df-hf-search" onSubmit={doSearch}>
                    <Search size={16} aria-hidden="true" />
                    <input ref={searchRef} type="search" aria-label="Search Hugging Face models or paste a model URL"
                        placeholder="Search models or paste a Hugging Face URL" value={query} onChange={(event) => setQuery(event.target.value)} />
                    <button type="submit">Search</button>
                </form>

                <div className="df-hf-body">
                    <div className="df-hf-results" aria-label="Hugging Face model results">
                        <div className="df-hf-section-title">Models {models.length ? <span>{models.length}</span> : null}</div>
                        {searchError && <p className="df-hf-error" role="alert">{searchError}</p>}
                        {!searchBusy && !models.length && !searchError && <p className="df-hf-empty">No models found. Try a model ID or URL.</p>}
                        {models.map((model) => <button key={model.id} type="button" className="df-hf-result" aria-selected={selectedId === model.id}
                            onClick={() => showPreview(model.id)}>
                            <strong>{model.id}</strong>
                            <span>{[model.pipelineTag, formatCount(model.downloads) && `${formatCount(model.downloads)} downloads`].filter(Boolean).join(' · ') || 'Open config'}</span>
                        </button>)}
                        {searchBusy && <p className="df-hf-loading"><LoaderCircle size={14} className="df-hf-spinner" /> Loading models…</p>}
                        {cursor && !searchBusy && <button type="button" className="df-hf-more" onClick={() => doSearch(null, cursor)}>Load more models <ArrowRight size={13} /></button>}
                    </div>

                    <div className="df-hf-preview" aria-live="polite">
                        {!selectedId && <div className="df-hf-placeholder"><span>HF</span><strong>Select a model</strong><p>Read its public config.json and check whether DeepFlow can represent it.</p></div>}
                        {previewBusy && <p className="df-hf-loading"><LoaderCircle size={16} className="df-hf-spinner" /> Reading config.json…</p>}
                        {previewError && <p className="df-hf-error" role="alert">{previewError}</p>}
                        {preview && <>
                            <div className="df-hf-preview-head">
                                <div><span className="df-hf-eyebrow">{inspection.family}</span><h3>{preview.repoId}</h3><span className="df-hf-commit">{preview.requestedRevision} · {preview.commit.slice(0, 12)}</span></div>
                                <span className={`df-hf-status ${inspection.supported ? 'is-supported' : ''}`}>{inspection.supported ? 'Supported' : 'Unsupported'}</span>
                            </div>
                            {!inspection.supported ? <p className="df-hf-unsupported">{inspection.reason}</p> : <>
                                <div className="df-hf-facts">
                                    <div><span>Width</span><strong>{inspection.configuration.dModel}</strong></div>
                                    <div><span>Blocks</span><strong>{inspection.configuration.nLayers}</strong></div>
                                    <div><span>Heads</span><strong>{inspection.configuration.nHeads}</strong></div>
                                    <div><span>Context</span><strong>{inspection.configuration.context}</strong></div>
                                </div>
                                <div className="df-hf-section-title">Editable components</div>
                                <ul className="df-hf-components">{inspection.components.map((component) => <li key={component.name}><strong>{component.name}</strong><span>{component.detail}</span></li>)}</ul>
                                <div className="df-hf-limits"><strong>Import scope</strong>{inspection.limitations.map((item) => <p key={item}>{item}</p>)}</div>
                            </>}
                            <a className="df-hf-external" href={preview.url} target="_blank" rel="noopener noreferrer">View source revision <ExternalLink size={12} /></a>
                        </>}
                    </div>
                </div>

                <footer className="df-hf-footer">
                    <span>Import replaces the current graph and clears training. Public configs only; no model code or weights are loaded.</span>
                    {importError && <span className="df-hf-error" role="alert">{importError}</span>}
                    <button type="button" className="df-hf-import" onClick={confirmImport} disabled={!inspection?.supported || importBusy}>
                        {importBusy ? 'Importing…' : 'Import architecture'} <ArrowRight size={14} />
                    </button>
                </footer>
            </section>
        </div>,
        document.body,
    );
}
