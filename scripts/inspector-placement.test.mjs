import assert from 'node:assert/strict';
import { test } from 'node:test';
import { placeInspector } from '../src/components/architecture/inspectorPlacement.js';

const bounds = { width: 1200, height: 760 };
const panel = { width: 272, height: 320 };
const overlap = (a, b) => a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;

test('inspector stays beside its node when there is empty space', () => {
    const anchor = { x: 410, y: 210, width: 188, height: 56 };
    const result = placeInspector({ anchor, panel, bounds });
    assert.equal(result.side, 'right');
    assert.equal(result.x, 616);
    assert.equal(result.y, 210);
    assert.equal(overlap(result, anchor), false);
});

test('inspector avoids a right palette and visible neighboring nodes', () => {
    const anchor = { x: 680, y: 160, width: 188, height: 56 };
    const palette = { x: 944, y: 16, width: 240, height: 630 };
    const neighbor = { x: 698, y: 310, width: 188, height: 56 };
    const result = placeInspector({ anchor, panel, bounds, obstacles: [palette], nodes: [neighbor] });
    assert.equal(result.side, 'left');
    assert.equal(overlap(result, palette), false);
    assert.equal(overlap(result, anchor), false);
    assert.equal(overlap(result, neighbor), false);
});

test('inspector remains inside the canvas when its node is near a corner or offscreen', () => {
    for (const anchor of [{ x: 5, y: 700, width: 188, height: 56 }, { x: 1150, y: -200, width: 188, height: 56 }]) {
        const result = placeInspector({ anchor, panel, bounds });
        assert.ok(result.x >= 12 && result.y >= 12);
        assert.ok(result.x + result.width <= bounds.width - 12);
        assert.ok(result.y + result.height <= bounds.height - 12);
    }
});
