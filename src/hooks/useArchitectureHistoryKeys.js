import { useEffect } from 'react';
import { useArchitectureHistory } from '../stores/architectureHistory.js';
import { architectureHistoryCommand } from './architectureHistoryKeys.js';

/** One listener for the architecture workspace, including both split panes. */
export function useArchitectureHistoryKeys() {
    useEffect(() => {
        const onKeyDown = (event) => {
            const command = architectureHistoryCommand(event);
            if (!command) return;
            event.preventDefault();
            useArchitectureHistory.getState()[command]();
        };
        // Inspector/popover controls stop bubbling keys to protect the canvas.
        // Capture still permits document Undo after a select/checkbox change;
        // text editing is excluded by architectureHistoryCommand above.
        window.addEventListener('keydown', onKeyDown, true);
        return () => window.removeEventListener('keydown', onKeyDown, true);
    }, []);
}
