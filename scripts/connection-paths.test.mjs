import assert from 'node:assert/strict';
import { test } from 'node:test';
import { connectionPath } from '../src/components/architecture/connectionPaths.js';

const path = (overrides = {}) => connectionPath({ sourceX: 100, sourceY: 100, targetX: 320, targetY: 280, ...overrides });

test('ordinary connections remain within their endpoints and keep straight port leads', () => {
    const points = path().match(/-?\d+(?:\.\d+)?/g).map(Number);
    assert.deepEqual(points.slice(0, 2), [100, 100]);
    assert.deepEqual(points.slice(-2), [320, 280]);
    assert.equal(points[2], 100, 'The output lead stays on the port axis');
    assert.ok(points[3] >= 121, 'The lead clears the output shape label before curving');
    assert.equal(points.at(-4), 320, 'The input lead stays on the port axis');
    assert.ok(points.at(-3) <= 259, 'The curve ends before the input shape label');
    for (const targetY of [100.01, 130, 280, 1000]) {
        const values = path({ targetY }).match(/-?\d+(?:\.\d+)?/g).map(Number);
        for (let i = 0; i < values.length; i += 2) {
            assert.ok(values[i] >= 100 && values[i] <= 320);
            assert.ok(values[i + 1] >= 100 && values[i + 1] <= targetY);
        }
    }
});

test('residual lanes clear 240px cards without a 190px excursion', () => {
    const wire = path({ targetX: 100, targetY: 600, detour: true });
    const values = wire.match(/-?\d+(?:\.\d+)?/g).map(Number);
    const xs = values.filter((_, index) => index % 2 === 0);
    assert.equal(Math.max(...xs), 244);
    assert.ok(Math.max(...xs) > 100 + 240 / 2);
    assert.ok(wire.startsWith('M 100 100'));
    assert.ok(wire.endsWith('L 100 600'));
});

test('close, reversed and coincident endpoints yield finite paths without an abrupt sideways fold', () => {
    for (const targetX of [100, 320, -40]) {
        for (const targetY of [99.99, 100, 100.01, -200]) {
            for (const detour of [false, true]) {
                const wire = path({ targetX, targetY, detour });
                assert.doesNotMatch(wire, /NaN|Infinity/);
                assert.ok(wire.startsWith('M 100 100'));
                assert.ok(wire.endsWith(`L ${targetX} ${targetY}`));
            }
        }
    }
    const almostLevel = path({ targetX: 100, targetY: 99.99 });
    const xs = almostLevel.match(/-?\d+(?:\.\d+)?/g).map(Number).filter((_, index) => index % 2 === 0);
    assert.ok(Math.max(...xs) - 100 < 1, 'Crossing endpoint height does not jump to a full lane');
});
