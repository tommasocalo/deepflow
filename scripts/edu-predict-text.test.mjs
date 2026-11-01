import assert from 'node:assert/strict';
import { test } from 'node:test';

// Prediction chrome must not cue a lesson answer (review r1: "for example B, 8, 28, 28" was M4 s2's answer).
const { MODULES } = await import('../src/edu/lessons/modules/index.js');
const UI = (await import('../src/edu/lessons/engine/text/en/ui.js')).default;
const { parseShapeInput, sameShape } = await import('../src/edu/lessons/engine/nodes.js');

const shapesIn = (text) => [...String(text).matchAll(/\bB(?:,\s*\d+)+/g)].map((match) => parseShapeInput(match[0]));

test('the shape examples in the prediction help match no expected or diagnosed lesson shape', () => {
    const examples = ['predict.shapeHelp', 'predict.invalidShape'].flatMap((key) => shapesIn(UI[key]));
    assert.ok(examples.length >= 2, 'both texts give an example shape');
    const lessonShapes = MODULES.flatMap((module) => module.steps.flatMap((step) => {
        if (step.predict?.kind !== 'shape') return [];
        return [step.predict.expectedShape, ...(step.predict.diagnose ?? []).map((rule) => rule.equals)].map((shape) => ({ where: `${module.id}.${step.id}`, shape }));
    }));
    assert.ok(lessonShapes.length > 0);
    for (const example of examples) {
        for (const { where, shape } of lessonShapes) assert.ok(!sameShape(example, shape), `${JSON.stringify(example)} is an answer of ${where}`);
    }
});
