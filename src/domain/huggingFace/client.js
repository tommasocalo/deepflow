/** Read-only Hugging Face Hub client for metadata, config and bounded ONNX graphs. */
const HUB = 'https://huggingface.co';
const REPO_PART = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const SHA = /^[a-f0-9]{40,64}$/i;
export const MAX_HUB_ONNX_BYTES = 64 * 1024 * 1024;

export class HubRequestError extends Error {
    constructor(message, status = null) {
        super(message);
        this.name = 'HubRequestError';
        this.status = status;
    }
}

export function validModelId(value) {
    const parts = typeof value === 'string' ? value.split('/') : [];
    return (parts.length === 1 || parts.length === 2)
        && parts.every((part) => REPO_PART.test(part) && part !== '.' && part !== '..');
}

/** Recognize a model id or a Hub model URL without trusting any external host. */
export function parseModelReference(input) {
    const text = String(input ?? '').trim();
    if (!text) return null;
    if (!/^https?:\/\//i.test(text)) return validModelId(text) ? { repoId: text, revision: 'main' } : null;
    let url;
    try { url = new URL(text); } catch { return null; }
    if (url.protocol !== 'https:' || url.hostname !== 'huggingface.co' || url.port) return null;
    let parts;
    try { parts = url.pathname.split('/').filter(Boolean).map((part) => decodeURIComponent(part)); }
    catch { return null; }
    if (parts.length < 1) return null;
    const marker = parts.findIndex((part) => ['tree', 'blob', 'resolve'].includes(part));
    const repoParts = marker < 0 ? parts : parts.slice(0, marker);
    const repoId = repoParts.join('/');
    if (!validModelId(repoId)) return null;
    const revision = marker >= 0 ? (parts[marker + 1] || 'main') : 'main';
    return /^[A-Za-z0-9._-]+$/.test(revision) ? { repoId, revision } : null;
}

async function jsonRequest(url, { signal, fetchImpl = fetch } = {}) {
    let response;
    try { response = await fetchImpl(url, { signal, headers: { Accept: 'application/json' } }); }
    catch (error) {
        if (error?.name === 'AbortError') throw error;
        throw new HubRequestError('Hugging Face is unreachable. Check the connection and try again.');
    }
    if (!response.ok) {
        const message = response.status === 404 ? 'Model or requested file not found on Hugging Face.'
            : [401, 403].includes(response.status) ? 'This repository is private or gated. Public access is required.'
                : response.status === 429 ? 'Hugging Face rate limit reached. Try again shortly.'
                    : `Hugging Face returned HTTP ${response.status}.`;
        throw new HubRequestError(message, response.status);
    }
    try { return { value: await response.json(), response }; }
    catch { throw new HubRequestError('Hugging Face returned an invalid JSON response.'); }
}

function nextCursor(link) {
    if (!link) return null;
    const next = link.split(',').find((part) => /rel="?next"?/i.test(part));
    const match = next?.match(/<([^>]+)>/);
    if (!match) return null;
    try {
        const url = new URL(match[1], HUB);
        return url.origin === HUB ? url.searchParams.get('cursor') : null;
    } catch { return null; }
}

/** Search public model repositories; `cursor` is the Hub's pagination token. */
export async function searchHubModels(query, { cursor = null, limit = 20, signal, fetchImpl } = {}) {
    const term = String(query ?? '').trim();
    if (!term) return { models: [], nextCursor: null };
    const url = new URL('/api/models', HUB);
    url.searchParams.set('search', term);
    url.searchParams.set('sort', 'downloads');
    url.searchParams.set('direction', '-1');
    url.searchParams.set('limit', String(Math.max(1, Math.min(30, Math.floor(limit)))));
    if (cursor) url.searchParams.set('cursor', cursor);
    const { value, response } = await jsonRequest(url.toString(), { signal, fetchImpl });
    if (!Array.isArray(value)) throw new HubRequestError('Hugging Face returned an unexpected search result.');
    return {
        models: value.filter((item) => validModelId(item.id)).map((item) => ({
            id: item.id,
            downloads: Number.isFinite(item.downloads) ? item.downloads : null,
            pipelineTag: item.pipeline_tag ?? null,
            lastModified: item.lastModified ?? null,
            private: item.private === true,
            gated: !!item.gated,
        })),
        nextCursor: nextCursor(response.headers?.get('Link')),
    };
}

/** Resolve the requested ref once; all subsequent downloads use this commit. */
export async function fetchHubModelMetadata(reference, { signal, fetchImpl } = {}) {
    const ref = typeof reference === 'string' ? parseModelReference(reference) : reference;
    if (!ref || !validModelId(ref.repoId)) throw new HubRequestError('Enter a valid Hugging Face model ID or URL.');
    const revision = ref.revision || 'main';
    if (!/^[A-Za-z0-9._-]+$/.test(revision)) throw new HubRequestError('The requested revision is invalid.');
    const idPath = ref.repoId.split('/').map(encodeURIComponent).join('/');
    const detailUrl = `${HUB}/api/models/${idPath}/revision/${encodeURIComponent(revision)}`;
    const { value: metadata } = await jsonRequest(detailUrl, { signal, fetchImpl });
    if (!metadata || !SHA.test(metadata.sha)) throw new HubRequestError('The model revision could not be pinned to a commit.');
    if (metadata.private || metadata.gated) throw new HubRequestError('This repository is private or gated. Public access is required.');
    return {
        repoId: validModelId(metadata.id) ? metadata.id : ref.repoId,
        requestedRevision: revision,
        commit: metadata.sha,
        files: (Array.isArray(metadata.siblings) ? metadata.siblings : []).filter((file) => validHubFilePath(file.rfilename)).map((file) => ({
            path: file.rfilename, size: Number.isSafeInteger(file.size) && file.size >= 0 ? file.size
                : Number.isSafeInteger(file.lfs?.size) && file.lfs.size >= 0 ? file.lfs.size : null,
        })),
        metadata: {
            downloads: Number.isFinite(metadata.downloads) ? metadata.downloads : null,
            pipelineTag: metadata.pipeline_tag ?? null,
            lastModified: metadata.lastModified ?? null,
            author: metadata.author ?? null,
        },
        url: `${HUB}/${idPath}/tree/${metadata.sha}`,
    };
}

function validHubFilePath(path) {
    return typeof path === 'string' && path.length > 0 && !path.includes('\\')
        && [...path].every((char) => char.charCodeAt(0) >= 32 && char.charCodeAt(0) !== 127)
        && path.split('/').every((part) => part && part !== '.' && part !== '..');
}

function pinnedFileUrl(preview, path) {
    if (!validModelId(preview?.repoId) || !SHA.test(preview?.commit) || !validHubFilePath(path)) {
        throw new HubRequestError('The file must belong to a pinned Hugging Face revision.');
    }
    const repo = preview.repoId.split('/').map(encodeURIComponent).join('/');
    return `${HUB}/${repo}/resolve/${preview.commit}/${path.split('/').map(encodeURIComponent).join('/')}`;
}

/** Missing config is allowed only for the architecture bridge (ONNX-only repos). */
export async function fetchHubModelConfig(reference, { signal, fetchImpl, allowMissingConfig = false } = {}) {
    const preview = await fetchHubModelMetadata(reference, { signal, fetchImpl });
    let config;
    try { ({ value: config } = await jsonRequest(pinnedFileUrl(preview, 'config.json'), { signal, fetchImpl })); }
    catch (error) {
        if (allowMissingConfig && error.status === 404) return { ...preview, config: null };
        throw error;
    }
    if (!config || typeof config !== 'object' || Array.isArray(config)) throw new HubRequestError('This config.json is not a model configuration.');
    return { ...preview, config };
}

/** Download only a listed ONNX graph, cancelling the stream at the byte limit. */
export async function fetchHubOnnxFile(preview, path, { signal, fetchImpl = fetch } = {}) {
    const file = preview?.files?.find((item) => item.path === path && /\.onnx$/i.test(item.path));
    if (!file) throw new HubRequestError('Choose an ONNX file listed in this model revision.');
    const sizeError = () => new HubRequestError('This ONNX graph exceeds the 64 MiB browser import limit. Export the graph with external tensor data, then load that smaller .onnx file.');
    if (file.size > MAX_HUB_ONNX_BYTES) throw sizeError();
    const url = pinnedFileUrl(preview, path);
    let response;
    try { response = await fetchImpl(url, { signal, headers: { Accept: 'application/octet-stream' } }); }
    catch (error) {
        if (error?.name === 'AbortError') throw error;
        throw new HubRequestError('The ONNX graph could not be downloaded from Hugging Face. Check the connection and try again.');
    }
    if (!response.ok) {
        throw new HubRequestError([401, 403].includes(response.status)
            ? 'This ONNX file is private or gated. Public access is required.'
            : `The ONNX download returned HTTP ${response.status}.`, response.status);
    }
    const length = Number(response.headers?.get('Content-Length'));
    if (length > MAX_HUB_ONNX_BYTES) {
        await response.body?.cancel?.();
        throw sizeError();
    }
    if (!response.body?.getReader) throw new HubRequestError('The ONNX download did not provide a readable file.');
    const reader = response.body.getReader();
    const chunks = [];
    let total = 0;
    try {
        while (true) {
            if (signal?.aborted) throw new DOMException('The ONNX download was cancelled.', 'AbortError');
            const { value, done } = await reader.read();
            if (done) break;
            total += value.byteLength;
            if (total > MAX_HUB_ONNX_BYTES) throw sizeError();
            chunks.push(value);
        }
    } catch (error) {
        await reader.cancel().catch(() => {});
        if (error?.name === 'AbortError' || error instanceof HubRequestError) throw error;
        throw new HubRequestError('The ONNX download was interrupted. Try again.');
    } finally { reader.releaseLock(); }
    if (!total) throw new HubRequestError('The ONNX file is empty.');
    const bytes = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return bytes;
}
