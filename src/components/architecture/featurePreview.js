/** A bounded view of one feature vector. Each tile groups adjacent features. */
export function featurePreview(shape, featureAxis = shape?.length - 1) {
    if (!Array.isArray(shape) || shape.length < 2) return null;
    const features = shape[featureAxis];
    if (!Number.isInteger(features) || features < 1) return null;
    // Keep GPT-2's 768→3072 comparison at the same scale: 16 features/tile.
    // Larger vectors increase grouping in powers of two, with at most192 tiles.
    const groupSize = Math.max(16, 2 ** Math.ceil(Math.log2(features / 192)));
    const count = Math.ceil(features / groupSize);
    const columns = count > 96 ? 24 : count > 48 ? 16 : 12;
    const rows = Math.ceil(count / columns);
    return { features, groupSize, count, columns, rows, width: columns * 7 - 1, height: rows * 7 - 1 };
}
