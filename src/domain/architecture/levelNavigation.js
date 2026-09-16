import { resolveGraphLevel } from './operations.js';

export const sameLevelPath = (a, b) => Array.isArray(a) && Array.isArray(b) &&
    a.length === b.length && a.every((id, index) => id === b[index]);

/** Check the entire path: the resolver can fall back when any segment is invalid. */
export function isValidLevelPath(graph, templates, path) {
    if (!Array.isArray(path) || path.some((id) => typeof id !== 'string' || !id)) return false;
    const level = resolveGraphLevel(graph, path, templates);
    return level.segments.length === path.length && level.segments.every((segment, index) => segment.id === path[index]);
}

export function nearestOpenAncestor(path, openPaths) {
    for (let depth = path.length - 1; depth > 0; depth -= 1) {
        const ancestor = openPaths.find((candidate) => sameLevelPath(candidate, path.slice(0, depth)));
        if (ancestor) return ancestor;
    }
    return [];
}

/** Retain insertion order and references so ordinary graph edits do not change navigation. */
export function reconcileOpenLevels(graph, templates, openPaths = [], currentPath = []) {
    const validPaths = [];
    for (const path of Array.isArray(openPaths) ? openPaths : []) {
        if (!path?.length || !isValidLevelPath(graph, templates, path)) continue;
        if (!validPaths.some((existing) => sameLevelPath(existing, path))) validPaths.push(path);
    }
    const active = isValidLevelPath(graph, templates, currentPath)
        ? currentPath
        : nearestOpenAncestor(Array.isArray(currentPath) ? currentPath : [], validPaths);
    return {
        openPaths: Array.isArray(openPaths) && validPaths.length === openPaths.length && validPaths.every((path, index) => path === openPaths[index]) ? openPaths : validPaths,
        currentPath: sameLevelPath(active, currentPath) ? currentPath : active,
    };
}
