/** Collections remain distinct from tensor dimensions until an explicit reduction. */
export const isShapeCollection = (shape) => shape?.kind === 'collection' && Array.isArray(shape.items);
export const shapeCollection = (items) => ({ kind: 'collection', items });
export function flattenShapeInputs(inputs) {
    return inputs.flatMap((shape) => isShapeCollection(shape) ? flattenShapeInputs(shape.items) : [shape]);
}
