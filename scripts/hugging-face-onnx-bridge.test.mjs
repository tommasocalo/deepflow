import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fetchHubOnnxFile, MAX_HUB_ONNX_BYTES } from '../src/domain/huggingFace/client.js';
import { prepareHubModel, selectHubOnnxFile, importHuggingFaceArchitecture } from '../src/domain/huggingFace/index.js';
import { validate } from '../src/validation/validator.js';

const commit = '607a30d783dfa663caf39e06633721c8d4cfcd7e';
const repoId = 'example/vision-component';
const fixture = readFileSync(new URL('./fixtures/tiny-linear-relu.onnx', import.meta.url));
const config = { model_type: 'vision-encoder-decoder', architectures: ['VisionEncoderDecoderModel'] };
const preview = { repoId, commit, requestedRevision: 'release-v1', config,
    files: [{ path: 'onnx/model.onnx', size: fixture.byteLength }] };

function mockHub({ modelConfig = config, files = preview.files, bytes = fixture, status = 200 } = {}) {
    const requests = [];
    const fetchImpl = async (url) => {
        requests.push(url);
        if (url.includes('/api/models/')) return Response.json({ id: repoId, sha: commit,
            siblings: files.map((file) => ({ rfilename: file.path, size: file.size })) });
        if (url.endsWith('/config.json')) return modelConfig === null
            ? new Response(null, { status: 404 }) : Response.json(modelConfig);
        return new Response(bytes, { status });
    };
    return { requests, fetchImpl };
}

function fakeStores() {
    let state = { graph: { nodes: { existing: { id: 'existing' } }, edges: [] }, currentPath: ['old'], openPaths: [['old']] };
    let trainingResets = 0;
    return {
        architectureStore: { setState: (patch) => { state = { ...state, ...patch }; }, getState: () => state },
        trainingStore: { getState: () => ({ resetGraph: () => { trainingResets += 1; } }) },
        getTrainingResets: () => trainingResets,
    };
}

test('unsupported Hub config uses its published ONNX graph pinned to the same immutable commit', async () => {
    const hub = mockHub();
    const result = await prepareHubModel({ repoId, revision: 'release-v1' }, hub);
    assert.equal(result.adapterId, 'onnx');
    assert.equal(result.inspection.supported, true);
    assert.equal(result.onnx.filePath, 'onnx/model.onnx');
    assert.equal(result.onnx.operatorGraph.nodes.length, 2);
    assert.equal(result.requiresOnnxSelection, false);
    assert.equal(result.inspection.components.length, 2);
    assert.deepEqual(hub.requests, [
        `https://huggingface.co/api/models/${repoId}/revision/release-v1`,
        `https://huggingface.co/${repoId}/resolve/${commit}/config.json`,
        `https://huggingface.co/${repoId}/resolve/${commit}/onnx/model.onnx`,
    ]);
    const stores = fakeStores();
    const source = importHuggingFaceArchitecture(result, stores);
    assert.equal(source.provider, 'huggingface');
    assert.equal(source.intermediateFormat, 'onnx');
    assert.equal(source.commit, commit);
    assert.equal(source.repoId, repoId);
    assert.equal(source.filePath, 'onnx/model.onnx');
    assert.equal(source.weightsLoaded, false);
    assert.equal(source.operatorGraph.nodes.length, 2);
    assert.deepEqual(stores.architectureStore.getState().currentPath, []);
    assert.equal(stores.getTrainingResets(), 1);
    const graph = stores.architectureStore.getState().graph;
    assert.equal(Object.values(graph.nodes).filter((node) => node.onnxSource.nodeIds.length).length, 2);
    assert.equal(validate(graph, { templates: {} }).ok, true);
});

test('ONNX-only Hub repositories import without config.json', async () => {
    const hub = mockHub({ modelConfig: null });
    const result = await prepareHubModel(repoId, hub);
    assert.equal(result.config, null);
    assert.equal(result.inspection.supported, true);
    assert.equal(result.adapterId, 'onnx');
});

test('supported config adapters remain preferred over ONNX downloads', async () => {
    const hub = mockHub({ modelConfig: { model_type: 'gpt2', architectures: ['GPT2LMHeadModel'],
        vocab_size: 31, n_positions: 4, n_embd: 8, n_layer: 2, n_head: 2,
        activation_function: 'gelu_new' } });
    const result = await prepareHubModel(repoId, hub);
    assert.equal(result.inspection.supported, true);
    assert.equal(result.adapterId, 'gpt2');
    assert.equal(result.onnx, null);
    assert.equal(hub.requests.length, 2);
});

test('multiple encoder/decoder or quantized graph files require explicit choice', async () => {
    const hub = mockHub({ files: [{ path: 'onnx/encoder.onnx' }, { path: 'onnx/decoder.onnx' }] });
    const result = await prepareHubModel(repoId, hub);
    assert.equal(result.requiresOnnxSelection, true);
    assert.equal(result.inspection.supported, false);
    assert.equal(result.onnx, null);
    assert.equal(hub.requests.length, 2);
    const selected = await selectHubOnnxFile(result, 'onnx/encoder.onnx', hub);
    assert.equal(selected.requiresOnnxSelection, false);
    assert.equal(selected.onnx.filePath, 'onnx/encoder.onnx');
    assert.match(selected.inspection.limitations.join(' '), /Imports only onnx\/encoder.onnx/);
    assert.equal(hub.requests.at(-1), `https://huggingface.co/${repoId}/resolve/${commit}/onnx/encoder.onnx`);
});

test('missing ONNX explains the unavailable bridge without changing the architecture', async () => {
    const hub = mockHub({ files: [{ path: 'model.safetensors' }] });
    const result = await prepareHubModel(repoId, hub);
    assert.equal(result.inspection.supported, false);
    assert.match(result.inspection.reason, /no published ONNX graph/);
    const stores = fakeStores();
    const previous = stores.architectureStore.getState();
    assert.throws(() => importHuggingFaceArchitecture(result, stores));
    assert.equal(stores.architectureStore.getState(), previous);
    assert.equal(stores.getTrainingResets(), 0);
});

test('unmapped ONNX operators do not become a successful or partial Hub import', async () => {
    // Custom operators must never be guessed as ordinary layers.
    const hub = mockHub();
    const result = await prepareHubModel(repoId, hub);
    result.onnx.operatorGraph.nodes[0].domain = 'custom.training';
    const stores = fakeStores();
    const previous = stores.architectureStore.getState();
    assert.throws(() => importHuggingFaceArchitecture(result, stores), /Custom domain/);
    assert.equal(stores.architectureStore.getState(), previous);
});

test('graph downloads reject unlisted paths and traversal before making requests', async () => {
    let called = false;
    const fetchImpl = async () => { called = true; };
    await assert.rejects(fetchHubOnnxFile(preview, 'other.onnx', { fetchImpl }), /listed/);
    await assert.rejects(fetchHubOnnxFile({ ...preview, files: [{ path: '../model.onnx' }] }, '../model.onnx', { fetchImpl }), /pinned/);
    assert.equal(called, false);
});

test('oversized metadata and response headers abort before reading a graph', async () => {
    let called = false;
    await assert.rejects(fetchHubOnnxFile({ ...preview, files: [{ path: 'onnx/model.onnx', size: MAX_HUB_ONNX_BYTES + 1 }] }, 'onnx/model.onnx', {
        fetchImpl: async () => { called = true; },
    }), /64 MiB/);
    assert.equal(called, false);
    let cancelled = false;
    const body = new ReadableStream({ cancel() { cancelled = true; } });
    await assert.rejects(fetchHubOnnxFile(preview, 'onnx/model.onnx', { fetchImpl: async () =>
        new Response(body, { headers: { 'Content-Length': String(MAX_HUB_ONNX_BYTES + 1) } }) }), /64 MiB/);
    assert.equal(cancelled, true);
});

test('streamed ONNX download enforces the size cap without Content-Length', async () => {
    let cancelled = false;
    let chunks = 0;
    const block = new Uint8Array(1024 * 1024);
    const body = new ReadableStream({
        pull(controller) { chunks += 1; controller.enqueue(block); },
        cancel() { cancelled = true; },
    });
    await assert.rejects(fetchHubOnnxFile(preview, 'onnx/model.onnx', { fetchImpl: async () => new Response(body) }), /64 MiB/);
    assert.equal(cancelled, true);
    assert.ok(chunks <= 66, 'stop downloading immediately after the 64 MiB limit');
});

test('HTTP failures, empty/malformed ONNX and cancellation remain explicit errors', async () => {
    for (const status of [403, 404, 429, 500]) {
        await assert.rejects(fetchHubOnnxFile(preview, 'onnx/model.onnx', {
            fetchImpl: async () => new Response(null, { status }),
        }), (error) => error.status === status);
    }
    await assert.rejects(prepareHubModel(repoId, mockHub({ bytes: new Uint8Array() })), /empty/);
    await assert.rejects(prepareHubModel(repoId, mockHub({ bytes: new TextEncoder().encode('not an ONNX model') })), /ONNX/);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(fetchHubOnnxFile(preview, 'onnx/model.onnx', {
        signal: controller.signal, fetchImpl: async () => new Response(fixture),
    }), { name: 'AbortError' });
});
