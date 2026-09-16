import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createInspectorPlacementScheduler, placeInspector } from '../src/components/architecture/inspectorPlacement.js';

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

function placementHarness(initialNodes = [{ id: 'inspected', dragging: false, x: 100 }]) {
    let nodes = initialNodes;
    let nextFrame = 0;
    const frames = new Map();
    const measurements = [];
    const scheduler = createInspectorPlacementScheduler({
        getNodes: () => nodes,
        position: () => measurements.push(structuredClone(nodes)),
        requestFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; },
        cancelFrame(id) { frames.delete(id); },
    });
    return {
        ...scheduler, measurements, frames,
        update(next) { nodes = next; scheduler.schedule(); },
        tick() {
            const pending = [...frames.values()];
            frames.clear();
            pending.forEach((callback) => callback());
        },
    };
}

test('dragging any node freezes placement, cancels queued work, and measures the released graph once', () => {
    const placement = placementHarness();
    placement.flush();
    placement.schedule();
    assert.equal(placement.frames.size, 1);
    for (const x of [120, 250, 400]) {
        placement.update([{ id: 'inspected', x: 100 }, { id: 'other', dragging: true, x }]);
        placement.schedule(); // Resize and mutation notifications during a drag.
        placement.tick();
    }
    assert.equal(placement.measurements.length, 1, 'No geometry reads during another node’s drag');
    assert.equal(placement.frames.size, 0);
    placement.update([{ id: 'inspected', x: 100 }, { id: 'other', dragging: false, x: 400 }]);
    placement.schedule();
    placement.schedule();
    assert.equal(placement.frames.size, 1);
    placement.tick();
    assert.equal(placement.measurements.length, 2);
    assert.equal(placement.measurements.at(-1)[1].x, 400);
});

test('a selection drag stays frozen until every dragged node is released', () => {
    const placement = placementHarness([{ id: 'a', dragging: true }, { id: 'b', dragging: true }]);
    placement.flush();
    placement.update([{ id: 'a', dragging: false }, { id: 'b', dragging: true }]);
    placement.tick();
    assert.equal(placement.measurements.length, 0);
    placement.update([{ id: 'a', dragging: false }, { id: 'b', dragging: false }]);
    placement.tick();
    assert.equal(placement.measurements.length, 1);
});

test('opening mid-drag defers initial measurement and recovers when reset replaces dragged nodes', () => {
    const placement = placementHarness([{ id: 'old', dragging: true, x: 900 }]);
    placement.flush();
    placement.schedule();
    assert.equal(placement.measurements.length, 0);
    assert.equal(placement.frames.size, 0);
    placement.update([{ id: 'restored', x: 100 }]);
    placement.tick();
    assert.deepEqual(placement.measurements, [[{ id: 'restored', x: 100 }]]);
    // Normal resize / pan / content changes remain responsive after cancellation.
    placement.schedule();
    placement.tick();
    assert.equal(placement.measurements.length, 2);
});

test('placement checks current dragging state when a queued frame runs and disposes pending work', () => {
    const nodes = [{ id: 'inspected', dragging: false }];
    const placement = placementHarness(nodes);
    placement.schedule();
    nodes[0].dragging = true;
    placement.tick();
    assert.equal(placement.measurements.length, 0);
    nodes[0].dragging = false;
    placement.schedule();
    placement.dispose();
    placement.schedule();
    placement.flush();
    placement.tick();
    assert.equal(placement.frames.size, 0);
    assert.equal(placement.measurements.length, 0);
});
