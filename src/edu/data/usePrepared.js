import { useEffect, useReducer, useState } from 'react';
import { prepareDataset, prepareDatasetSync, preparedKey, subscribePrepared } from './prepare.js';

/**
 * The PreparedDataset for a browser dataset and its options, split and seed (data/prepare.js). Toy sets and Iris
 * prepare synchronously; digits loads its data chunk once, and the component re-renders when it is ready.
 * Export-only presets return `{ prepared: null }`.
 */
export function usePrepared(spec, options, split, dataSeed) {
    const key = spec?.training === 'browser' ? preparedKey(spec, options, split, dataSeed) : null;
    const [, refresh] = useReducer((count) => count + 1, 0);
    const [failure, setFailure] = useState(null);
    useEffect(() => subscribePrepared(() => refresh()), []);
    const prepared = key ? prepareDatasetSync(spec, options, split, dataSeed) : null;
    const missing = !!key && !prepared;
    useEffect(() => {
        if (!missing) return undefined;
        let cancelled = false;
        prepareDataset(spec, options, split, dataSeed).then(
            () => { if (!cancelled) refresh(); },
            (error) => { if (!cancelled) setFailure({ key, message: error?.message ?? String(error) }); },
        );
        return () => { cancelled = true; };
        // The key covers spec, options, split and seed.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [key, missing]);
    const error = failure?.key === key ? failure.message : null;
    return { prepared, loading: missing && !error, error };
}
