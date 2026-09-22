import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';
import { completeVariableReference, variableReferenceAtCaret } from '../src/components/architecture/variableCompletion.js';

const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => errors.push(error));
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/', pretendToBeVisual: true, virtualConsole });
const globals = new Map();
for (const [name, value] of Object.entries({ window: dom.window, document: dom.window.document, navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, IS_REACT_ACT_ENVIRONMENT: true })) {
    globals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
}
const { act, createElement: h, useState } = await import('react');
const { createRoot } = await import('react-dom/client');
const directory = await mkdtemp(join(dirname(dirname(fileURLToPath(import.meta.url))), '.variable-input-'));
let ParameterField;
try {
    const output = join(directory, 'field.mjs');
    await build({ entryPoints: ['src/components/architecture/ParameterField.jsx'], outfile: output, bundle: true, packages: 'external', format: 'esm', platform: 'node', jsx: 'automatic', loader: { '.css': 'empty' }, logLevel: 'silent' });
    ParameterField = (await import(pathToFileURL(output).href)).default;
} finally {
    await rm(directory, { recursive: true, force: true });
}

let container;
let root;
let latest;
const suggestions = [
    { name: 'width', value: 768, owner: 'Transformer', type: 'int' },
    { name: 'weight_shape', value: [32, 16], owner: 'Projection', type: 'tuple' },
    { name: 'dropout', value: 0.1, owner: 'Attention', type: 'float' },
];
function Field({ initial = null, name = 'width', type = 'float' }) {
    const [value, setValue] = useState(initial);
    return h(ParameterField, { param: { name, type }, value, suggestions, onChange(next) { latest = next; setValue(next); } });
}
beforeEach(() => {
    errors.length = 0;
    latest = undefined;
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
});
afterEach(async () => {
    await act(() => root.unmount());
    container.remove();
    assert.deepEqual(errors, []);
});
after(() => {
    dom.window.close();
    for (const [name, descriptor] of globals) {
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else delete globalThis[name];
    }
});
const render = (props) => act(() => root.render(h(Field, props)));
const input = (label) => label ? container.querySelector(`[aria-label="${label}"]`) : container.querySelector('input');
const options = () => [...container.querySelectorAll('[role="option"]')];
const type = (field, text, caret = text.length) => act(() => {
    field.focus();
    Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value').set.call(field, text);
    field.setSelectionRange(caret, caret);
    field.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
});
async function key(field, value) {
    const event = new dom.window.KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true });
    await act(() => field.dispatchEvent(event));
    return event;
}

test('parameter completion locates and replaces only the reference around the caret', () => {
    const text = '2 * $width + $dropout';
    const reference = variableReferenceAtCaret(text, 7);
    assert.deepEqual(reference, { start: 4, end: 10, prefix: 'wi' });
    assert.deepEqual(completeVariableReference(text, reference, 'new_width'), { value: '2 * $new_width + $dropout', caret: 14 });
    assert.equal(variableReferenceAtCaret('2 * width', 9), null);
    assert.deepEqual(variableReferenceAtCaret('2 * $', 5), { start: 4, end: 5, prefix: '' });
});

test('typing $ offers scoped names, resolved values and owners; arrows and Enter insert a choice', async () => {
    await render();
    await type(input(), '$');
    assert.equal(input().getAttribute('aria-expanded'), 'true');
    assert.equal(options().length, 3);
    assert.match(options()[0].textContent, /\$width768Transformer/);
    assert.match(options()[1].textContent, /32 × 16Projection/);
    await key(input(), 'ArrowDown');
    assert.equal(options()[1].getAttribute('aria-selected'), 'true');
    await key(input(), 'Enter');
    assert.equal(latest, '$weight_shape');
    assert.equal(input().value, '$weight_shape');
    assert.equal(options().length, 0);
    assert.equal(input().selectionStart, '$weight_shape'.length);
});

test('prefix matching and caret insertion preserve the rest of the expression', async () => {
    await render();
    await type(input(), '2 * $wi + 4', 7);
    assert.equal(options().length, 1);
    await key(input(), 'Enter');
    assert.equal(latest, '2 * $width + 4');
    assert.equal(input().selectionStart, 10);
    await type(input(), '2 * $weight_shape + 4', 6);
    assert.equal(options().length, 2, 'A caret after $w offers both matching references');
    await key(input(), 'Enter');
    assert.equal(latest, '2 * $width + 4', 'Completion replaces the whole old token, not only its prefix');
});

test('Escape dismisses suggestions without changing the expression or swallowing later Enter', async () => {
    await render();
    await type(input(), '$dr');
    const escape = await key(input(), 'Escape');
    assert.equal(escape.defaultPrevented, true);
    assert.equal(latest, '$dr');
    assert.equal(options().length, 0);
    const enter = await key(input(), 'Enter');
    assert.equal(enter.defaultPrevented, false);
    await type(input(), '$dro');
    assert.equal(options().length, 1, 'Editing again reopens completion');
});

test('shape completion stays outside the scrolling axes and changes only the chosen dimension', async () => {
    await render({ initial: [32, '$wi'], name: 'shape', type: 'tuple' });
    await type(input('shape dimension 2'), '$wi');
    const list = container.querySelector('[role="listbox"]');
    assert.equal(list.parentElement.classList.contains('df-param-row'), true);
    assert.equal(list.closest('.df-shape-axes'), null);
    await act(() => options()[0].click());
    assert.deepEqual(latest, [32, '$width']);
    assert.equal(input('shape dimension 1').value, '32');
    assert.equal(input('shape dimension 2').value, '$width');
});

test('whole-shape references retain their scalar binding and numeric editing keeps decimal text until blur', async () => {
    await render({ initial: '$weight_shape', name: 'shape', type: 'tuple' });
    await type(input('shape'), '$we');
    await key(input('shape'), 'Enter');
    assert.equal(latest, '$weight_shape');
    assert.equal(input('shape dimension 1'), null);
    await act(() => root.render(h(Field, { key: 'number' })));
    await type(input(), '1.00e-2');
    assert.equal(latest, '1.00e-2');
    await act(() => input().blur());
    assert.equal(latest, 0.01);
});
