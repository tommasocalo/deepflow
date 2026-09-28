/** Read-only Hugging Face Hub client. Only JSON metadata and config are fetched. */
const HUB = 'https://huggingface.co';
const REPO_PART = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const SHA = /^[a-f0-9]{40,64}$/i;

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
    const parts = url.pathname.split('/').filter(Boolean).map((part) => decodeURIComponent(part));
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
        const message = response.status === 404 ? 'Model or config.json not found on Hugging Face.'
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

/** Resolve the requested ref to a commit before loading its config.json. */
export async function fetchHubModelConfig(reference, { signal, fetchImpl } = {}) {
    const ref = typeof reference === 'string' ? parseModelReference(reference) : reference;
    if (!ref || !validModelId(ref.repoId)) throw new HubRequestError('Enter a valid Hugging Face model ID or URL.');
    const revision = ref.revision || 'main';
    if (!/^[A-Za-z0-9._-]+$/.test(revision)) throw new HubRequestError('The requested revision is invalid.');
    const idPath = ref.repoId.split('/').map(encodeURIComponent).join('/');
    const detailUrl = `${HUB}/api/models/${idPath}/revision/${encodeURIComponent(revision)}`;
    const { value: metadata } = await jsonRequest(detailUrl, { signal, fetchImpl });
    if (!metadata || !SHA.test(metadata.sha)) throw new HubRequestError('The model revision could not be pinned to a commit.');
    if (metadata.private || metadata.gated) throw new HubRequestError('This repository is private or gated. Public access is required.');
    const configUrl = `${HUB}/${idPath}/resolve/${metadata.sha}/config.json`;
    const { value: config } = await jsonRequest(configUrl, { signal, fetchImpl });
    if (!config || typeof config !== 'object' || Array.isArray(config)) throw new HubRequestError('This config.json is not a model configuration.');
    return {
        repoId: metadata.id || ref.repoId,
        requestedRevision: revision,
        commit: metadata.sha,
        config,
        metadata: {
            downloads: Number.isFinite(metadata.downloads) ? metadata.downloads : null,
            pipelineTag: metadata.pipeline_tag ?? null,
            lastModified: metadata.lastModified ?? null,
            author: metadata.author ?? null,
        },
        url: `${HUB}/${idPath}/tree/${metadata.sha}`,
    };
}
