/**
 * Fitting table text into a column without changing what it says. A cut value
 * must never read as another value: numbers are kept whole, an expression is
 * cut where the ellipsis visibly belongs to it ("4 * d_…", never "4 …"), and a
 * cell that still does not fit drops whole trailing fields. Widths are counted
 * in characters of the table's monospace font.
 */

const NUMBER_END = /[\d.]+$/;
const NUMBER_START = /^[\d.]/;
const TOKEN_CHAR = /[\w.]/;

/**
 * Shorten an expression to at most `max` characters, ellipsis included. The cut
 * never splits a number and never ends on a complete number either, since
 * `4 * d_model` cut to `4…` would read as the value 4.
 */
export function cutExpression(text, max) {
    const value = String(text ?? '');
    if (value.length <= max) return value;
    let cut = value.slice(0, Math.max(0, max - 1));
    const next = value[cut.length] ?? '';
    // Inside a number: back off to its start.
    if (NUMBER_END.test(cut) && (NUMBER_START.test(next) || TOKEN_CHAR.test(next))) {
        const tokenStart = cut.search(/[\w.]+$/);
        cut = cut.slice(0, tokenStart);
    }
    cut = cut.trimEnd();
    // Ending on a whole number (not part of a name such as ln_1) would read as the value.
    const trailing = cut.match(/[\w.]+$/)?.[0] ?? '';
    if (trailing && /^[\d.]+$/.test(trailing)) cut = cut.slice(0, cut.length - trailing.length).trimEnd();
    return cut ? `${cut}…` : '…';
}

const REFERENCE = /^\$[A-Za-z_]\w*$/;

/** Approximate width of a part in monospace characters, including its margin. */
function partWidth(part, text, previous) {
    if (part.kind === 'field') {
        const prefix = part.prefix ? `${part.prefix}${/^[A-Za-z_]/.test(part.prefix) ? '=' : ''}` : '';
        return prefix.length + text.length + (previous?.kind === 'field' ? 1.1 : 0);
    }
    // Separators carry 3px margins each side; notes are a smaller sans-serif font.
    if (part.kind === 'note') return text.length * 0.8 + 1;
    return text.length + 1;
}

/**
 * The text each part of a config cell shows within `budget` characters.
 * `display(part)` gives a field's full text; `expression(part)` says whether it
 * may be cut inside. Returns { texts, hiddenFrom }: shortened texts by part
 * index, and the index of the first part dropped (an ellipsis replaces it).
 * In order of preference: shorten formulas (to at least 6 characters), drop
 * trailing groups (after a `·`, or a note), shorten `$name` references too,
 * and only then drop any trailing field. A pair such as `in → out` stays
 * together as long as possible.
 */
export function fitConfigParts(parts, { budget, display, expression }) {
    const full = parts.map((part) => part.kind === 'field' ? display(part) : part.text);
    const width = (texts, count) => texts.slice(0, count).reduce((sum, text, index) => sum + partWidth(parts[index], text, parts[index - 1]), 0);
    if (!Number.isFinite(budget) || budget <= 0 || width(full, parts.length) <= budget) return { texts: full, hiddenFrom: null };
    const attempt = (count, references) => {
        // Room for the "…" that stands for the dropped parts.
        const room = count < parts.length ? budget - 2 : budget;
        const texts = [...full];
        // At most one reference is cut: `$d_mod… → $voca…` says too little.
        let referenceCut = -1;
        for (let guard = 0; guard < count * 4 && width(texts, count) > room; guard++) {
            let longest = -1;
            for (let index = 0; index < count; index++) {
                if (parts[index].kind !== 'field' || !expression(parts[index]) || texts[index].length <= 6) continue;
                if (REFERENCE.test(full[index]) && (!references || (referenceCut >= 0 && referenceCut !== index))) continue;
                if (longest < 0 || texts[index].length > texts[longest].length) longest = index;
            }
            if (longest >= 0 && REFERENCE.test(full[longest])) referenceCut = longest;
            if (longest < 0) break;
            const over = Math.ceil(width(texts, count) - room);
            const shorter = cutExpression(full[longest], Math.max(6, texts[longest].length - over));
            if (shorter.length >= texts[longest].length) break;
            texts[longest] = shorter;
        }
        return width(texts, count) <= room ? { texts, hiddenFrom: count < parts.length ? count : null } : null;
    };
    // Where a cut leaves whole groups: before a `·` separator or a note.
    const groupEnds = [];
    const fieldEnds = [];
    for (let count = parts.length - 1; count >= 1; count--) {
        if (parts[count - 1].kind === 'text') continue;
        fieldEnds.push(count);
        if (parts[count].kind === 'note' || (parts[count].kind === 'text' && parts[count].text === '·')) groupEnds.push(count);
    }
    for (const references of [false, true]) {
        const result = attempt(parts.length, references) ?? groupEnds.reduce((found, count) => found ?? attempt(count, references), null);
        if (result) return result;
    }
    for (const count of fieldEnds) {
        const result = attempt(count, true);
        if (result) return result;
    }
    // Nothing fits: the first field, cut as far as it goes.
    const texts = [...full];
    if (parts[0]?.kind === 'field' && expression(parts[0])) texts[0] = cutExpression(full[0], 6);
    return { texts, hiddenFrom: parts.length > 1 ? 1 : null };
}

/**
 * Widths, in characters of the shape font, of the glyphs a shape label draws
 * with margins: the × between axes, the × after a collection count, and the
 * gap before a "known in 1/12" mark. One character each unless measured.
 */
export const PLAIN_GLYPHS = Object.freeze({ times: 1, collection: 1, mark: 1 });

/** Width of a shape label as drawn, in characters: `B×1024×768`, `12×[B×1024×64]`. */
export function drawnShapeLength(text, glyphs = PLAIN_GLYPHS) {
    if (!text) return 0;
    const collection = text.match(/^(\d+) tensors(?: \[(.*)\])?$/);
    if (collection) return collection[2] ? collection[1].length + glyphs.collection + 2 + drawnShapeLength(collection[2], glyphs) : text.length;
    const axes = text.split(' × ');
    return axes.join('').length + (axes.length - 1) * glyphs.times;
}

/**
 * A shape label within `budget` characters: whole axes only, so
 * `B×64…256×32×32` becomes `B×64…256×…`, never `B×64…256×3…`. An axis that
 * varies across instances says the most, so axes that do not give way first:
 * `B×1024×1536…3072` becomes `B×…×1536…3072`, not `B×1024×…`. A collection
 * (`12 tensors [B × 1024 × 64]`) is measured as drawn, `12×[B×1024×64]`,
 * and cut inside its brackets. `glyphs` gives the drawn width of separators.
 */
export function fitShapeText(text, budget, glyphs = PLAIN_GLYPHS) {
    if (!text || !Number.isFinite(budget) || budget <= 0) return text;
    const collection = text.match(/^(\d+) tensors(?: \[(.*)\])?$/);
    if (collection) {
        if (!collection[2]) return text;
        const inner = fitShapeText(collection[2], budget - collection[1].length - 2 - glyphs.collection, glyphs);
        return `${collection[1]} tensors [${inner}]`;
    }
    if (drawnShapeLength(text, glyphs) <= budget) return text;
    const axes = text.split(' × ');
    const width = (parts) => parts.join('').length + (parts.length - 1) * glyphs.times;
    const trailing = [...axes];
    while (trailing.length > 1 && width([...trailing, '…']) > budget) trailing.pop();
    const ranges = axes.map((axis, index) => axis.includes('…') ? index : -1).filter((index) => index > 0);
    // The trailing cut drops a range: keep the batch axis and the ranges, eliding fixed axes in between.
    if (ranges.length && ranges.at(-1) >= trailing.length) {
        const kept = [];
        axes.forEach((axis, index) => {
            const keep = index === 0 || ranges.includes(index);
            if (keep) kept.push(axis);
            else if (kept.at(-1) !== '…') kept.push('…');
        });
        if (width(kept) <= budget) return kept.join(' × ');
    }
    return `${trailing.join(' × ')} × …`;
}
