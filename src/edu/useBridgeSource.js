import { useEffect, useState } from 'react';
import { onSourceRegistered, readSource, subscribeSource } from './bridge.js';

/**
 * A bridge source's latest read() (docs/edu/SPEC.md §6.2) for React views: re-read on the source's notifications,
 * once right after mount, and when the source is registered after the view.
 */
export function useBridgeSource(name) {
    const [value, setValue] = useState(() => readSource(name));
    useEffect(() => {
        const refresh = () => setValue(readSource(name));
        let unsubscribe = subscribeSource(name, refresh);
        const stop = onSourceRegistered((registered) => {
            if (registered !== name) return;
            unsubscribe();
            unsubscribe = subscribeSource(name, refresh);
            refresh();
        });
        const initial = setTimeout(refresh, 0);
        return () => {
            clearTimeout(initial);
            unsubscribe();
            stop();
        };
    }, [name]);
    return value;
}
