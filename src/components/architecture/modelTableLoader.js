import { lazy } from 'react';

/**
 * The Model table is off by default, so its code loads on first use.
 * Hovering or focusing the Table toggle starts the load early, and the
 * focus request lets the panel take focus once it has mounted, however
 * long the load took.
 */
const load = () => import('./ModelTablePanel.jsx');
export const ModelTablePanel = lazy(load);
export const preloadModelTable = () => { load().catch(() => {}); };

const focusRequests = new Set();
/** Ask the table of a pane to focus its active cell when it opens. */
export const requestModelTableFocus = (paneId) => { focusRequests.add(paneId); };
/** True once per request: the panel focuses itself. */
export const takeModelTableFocus = (paneId) => focusRequests.delete(paneId);
