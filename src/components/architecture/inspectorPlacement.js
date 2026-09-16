const intersectionArea = (a, b) => Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x))
    * Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

/** Place a panel next to its node, preferring empty canvas over other nodes or UI. */
export function placeInspector({ anchor, panel, bounds, obstacles = [], nodes = [], margin = 12, gap = 18 }) {
    const width = Math.min(panel.width, Math.max(0, bounds.width - margin * 2));
    const height = Math.min(panel.height, Math.max(0, bounds.height - margin * 2));
    const maxX = Math.max(margin, bounds.width - margin - width);
    const maxY = Math.max(margin, bounds.height - margin - height);
    const middleX = anchor.x + anchor.width / 2;
    const middleY = anchor.y + anchor.height / 2;
    const ys = [anchor.y, middleY - height / 2, anchor.y + anchor.height - height];
    const candidates = [
        ...ys.map((y) => ({ x: anchor.x + anchor.width + gap, y, side: 'right' })),
        ...ys.map((y) => ({ x: anchor.x - gap - width, y, side: 'left' })),
        { x: middleX - width / 2, y: anchor.y + anchor.height + gap, side: 'below' },
        { x: middleX - width / 2, y: anchor.y - gap - height, side: 'above' },
        { x: margin, y: margin, side: 'corner' },
        { x: maxX, y: margin, side: 'corner' },
        { x: margin, y: maxY, side: 'corner' },
        { x: maxX, y: maxY, side: 'corner' },
    ].map((candidate) => ({ ...candidate, x: clamp(candidate.x, margin, maxX), y: clamp(candidate.y, margin, maxY), width, height }));
    const score = (candidate) => {
        const distance = Math.hypot(
            Math.max(candidate.x - anchor.x - anchor.width, 0, anchor.x - candidate.x - width),
            Math.max(candidate.y - anchor.y - anchor.height, 0, anchor.y - candidate.y - height),
        );
        return intersectionArea(candidate, anchor) * 30
            + obstacles.reduce((sum, obstacle) => sum + intersectionArea(candidate, obstacle), 0) * 20
            + nodes.reduce((sum, node) => sum + intersectionArea(candidate, node), 0)
            + distance * 20;
    };
    return candidates.reduce((best, candidate) => score(candidate) < score(best) ? candidate : best);
}
