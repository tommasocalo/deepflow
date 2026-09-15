import { getEntry } from '../domain/library/index.js';

/**
 * Validate that required parameters are set and that values are coercible
 * into their declared types. Returns an array of error messages keyed to
 * the parameter name.
 *
 * @param {import('../domain/architecture/types.js').LayerNode} node
 * @returns {{ paramName: string, message: string }[]}
 */
export function validateLayerParameters(node) {
    if (node.kind !== 'layer') return [];
    const entry = getEntry(node.libraryKey);
    if (!entry) return [{ paramName: '', message: `Unknown block "${node.libraryKey}"` }];

    /** @type {{ paramName: string, message: string }[]} */
    const issues = [];
    for (const spec of entry.parameters) {
        const v = node.parameters[spec.name];
        if (spec.required && (v === null || v === undefined || v === '')) {
            issues.push({ paramName: spec.name, message: `${spec.name} is required` });
            continue;
        }
        if (v === null || v === undefined || v === '') continue;
        if (spec.type === 'int' && !Number.isInteger(Number(v))) {
            issues.push({ paramName: spec.name, message: `${spec.name} must be an integer` });
        } else if (spec.type === 'float' && Number.isNaN(Number(v))) {
            issues.push({ paramName: spec.name, message: `${spec.name} must be a number` });
        } else if (spec.type === 'bool' && typeof v !== 'boolean') {
            issues.push({ paramName: spec.name, message: `${spec.name} must be true or false` });
        } else if (spec.type === 'enum' && spec.options && !spec.options.includes(v)) {
            issues.push({
                paramName: spec.name,
                message: `${spec.name} must be one of: ${spec.options.join(', ')}`,
            });
        }
    }
    return issues;
}
