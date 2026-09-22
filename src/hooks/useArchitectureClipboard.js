import { useEffect, useRef, useState } from 'react';
import { GRAPH_CLIPBOARD_MIME, readGraphClipboard } from '../domain/architecture/clipboard.js';
import { isEditingText } from './architectureHistoryKeys.js';

/** Browser clipboard events work with keyboard and native Copy/Paste menus. */
export function useArchitectureClipboard(store, active) {
    const [notice, setNotice] = useState('');
    const pasted = useRef(null);
    useEffect(() => {
        if (!active) return;
        const nativeField = (event) => event.defaultPrevented || isEditingText(event.target) || event.target?.closest?.('select');
        const copy = (event) => {
            if (nativeField(event) || !event.clipboardData || window.getSelection()?.toString()) return;
            const selection = store.getState().copySelection();
            if (!selection) return;
            const text = JSON.stringify(selection);
            event.clipboardData.setData(GRAPH_CLIPBOARD_MIME, text);
            event.clipboardData.setData('text/plain', text);
            event.preventDefault();
            pasted.current = null;
            setNotice(`${selection.nodes.length} ${selection.nodes.length === 1 ? 'node' : 'nodes'} copied`);
        };
        const paste = (event) => {
            if (nativeField(event) || !event.clipboardData) return;
            const text = event.clipboardData.getData(GRAPH_CLIPBOARD_MIME) || event.clipboardData.getData('text/plain');
            const selection = readGraphClipboard(text);
            if (!selection) return;
            event.preventDefault();
            const state = store.getState();
            const context = state.getMutationContext();
            const previous = pasted.current;
            const count = previous?.text === text && previous.levelKey === context.levelKey ? previous.count + 1 : 1;
            const result = state.pasteSelection(selection, { x: count * 36, y: count * 36 }, context);
            if (result.error) { setNotice(result.error); return; }
            pasted.current = { text, levelKey: context.levelKey, count };
            setNotice(`${result.nodeIds.length} ${result.nodeIds.length === 1 ? 'node' : 'nodes'} pasted`);
        };
        window.addEventListener('copy', copy);
        window.addEventListener('paste', paste);
        return () => { window.removeEventListener('copy', copy); window.removeEventListener('paste', paste); };
    }, [store, active]);
    useEffect(() => {
        if (!notice) return;
        const timer = setTimeout(() => setNotice(''), 3000);
        return () => clearTimeout(timer);
    }, [notice]);
    return active ? notice : '';
}
