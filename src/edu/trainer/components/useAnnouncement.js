import { useEffect, useRef, useState } from 'react';
import { runAnnouncement } from '../describe.js';

/** The text for a polite live region: set once when a run ends, never on every epoch. */
export function useRunAnnouncement(record) {
    const [text, setText] = useState('');
    const last = useRef(null);
    const key = record && ['completed', 'stopped', 'failed'].includes(record.status) ? `${record.id}:${record.status}` : null;
    useEffect(() => {
        if (!key || last.current === key) return;
        last.current = key;
        setText(runAnnouncement(record));
        // Only the finished state matters; the record object changes on every progress update.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [key]);
    return text;
}
