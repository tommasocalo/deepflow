import { useCallback } from 'react';
import { COLUMNS, CONFIG_COLUMN, fieldTargets } from './modelTableEdits.js';

const PAGE = 10;

/**
 * Keys of the treegrid while no editor is open: arrows, Home/End and Page keys
 * move the active cell (Ctrl/⌘ with Home/End to the first or last row); Right
 * and Left open and close rows; Enter and F2 edit a setting or open the row's
 * level; typing on a setting starts editing it; Escape returns to the canvas.
 */
export function useGridKeyboard({ rows, byKey, limit, editing, cells, context, actions, openRow, focusCell, setActive, setMessage, focusCanvas }) {
    return useCallback((event) => {
        if (editing || event.target.closest?.('.df-mt-editor')) return;
        const cellElement = event.target.closest?.('[data-col]');
        const entry = byKey.get(cellElement?.closest('tr')?.dataset.rowKey);
        if (!cellElement || !entry) return;
        const { row, index } = entry;
        const column = Number(cellElement.dataset.col);
        const rendered = Math.min(rows.length, limit);
        const go = (nextIndex, nextColumn = column) => {
            const target = rows[Math.max(0, Math.min(rendered - 1, nextIndex))];
            const targetColumn = Math.max(0, Math.min(COLUMNS.length - 1, nextColumn));
            setActive({ key: target.key, column: targetColumn });
            // An explanation belongs to the cell it was given for.
            setMessage('');
            focusCell(target.key, targetColumn);
        };
        const targets = Object.values(fieldTargets(row, cells, context));
        const editable = targets.some((target) => target.editable);
        // A setting that cannot be edited here says where to change it.
        const explain = () => setMessage(targets.map((target) => `${target.field.name}: ${target.hint}`).join(' · '));
        const modifier = event.ctrlKey || event.metaKey;
        switch (event.key) {
            case 'ArrowDown': go(index + 1); break;
            case 'ArrowUp': go(index - 1); break;
            case 'PageDown': go(index + PAGE); break;
            case 'PageUp': go(index - PAGE); break;
            case 'Home': if (modifier) go(0); else go(index, 0); break;
            case 'End': if (modifier) go(rendered - 1); else go(index, COLUMNS.length - 1); break;
            case 'ArrowRight':
                if (column === 0 && row.expandable && !row.expanded) actions.toggle(row);
                else go(index, column + 1);
                break;
            case 'ArrowLeft':
                if (column > 0) go(index, column - 1);
                else if (row.expanded) actions.toggle(row);
                else if (row.parentKey && byKey.has(row.parentKey)) go(byKey.get(row.parentKey).index, 0);
                break;
            case 'Enter':
                if (column === CONFIG_COLUMN && editable) actions.startEdit(row);
                else if (column === CONFIG_COLUMN && targets.length) explain();
                else openRow(row);
                break;
            case 'F2':
                if (editable) actions.startEdit(row);
                else if (targets.length) explain();
                break;
            case ' ':
                if (column === 0 && row.expandable) actions.toggle(row);
                break;
            case 'Escape':
                focusCanvas();
                break;
            default:
                // Spreadsheet style: typing on a setting starts editing it.
                if (column === CONFIG_COLUMN && editable && event.key.length === 1 && !modifier && !event.altKey) actions.startEdit(row, null, event.key);
                else return;
        }
        event.preventDefault();
    }, [rows, byKey, limit, editing, cells, context, actions, openRow, focusCell, setActive, setMessage, focusCanvas]);
}
