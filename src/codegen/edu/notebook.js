/**
 * train.py as a Jupyter notebook for Colab (docs/edu/SPEC.md §6.11.3): nbformat 4.5 with two code cells. Cell 1 is
 * the script without its `if __name__ == "__main__":` block (in a notebook `__name__` is "__main__", so keeping it
 * would train twice); cell 2 is `model = train()`.
 */

const MAIN_GUARD = /\n*if __name__ == "__main__":\n[\s\S]*$/;

/** The script without its trailing `if __name__ == "__main__":` block. */
export function stripMainGuard(script) {
    return String(script).replace(MAIN_GUARD, '\n');
}

/** Source text as nbformat's list of lines, each but the last ending in "\n". */
function sourceLines(text) {
    const lines = String(text).replace(/\n+$/, '').split('\n');
    return lines.map((line, index) => (index < lines.length - 1 ? `${line}\n` : line));
}

function codeCell(id, text) {
    return { cell_type: 'code', execution_count: null, id, metadata: {}, outputs: [], source: sourceLines(text) };
}

/**
 * @param {string} script train.py
 * @param {{ name?: string }} [options] `name` becomes the notebook's Colab title
 * @returns {object} the notebook as JSON-serialisable nbformat 4.5
 */
export function trainingNotebook(script, { name = 'train.ipynb' } = {}) {
    return {
        cells: [
            codeCell('train-script', stripMainGuard(script)),
            codeCell('run-training', 'model = train()'),
        ],
        metadata: {
            colab: { name, provenance: [] },
            kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
            language_info: { name: 'python' },
        },
        nbformat: 4,
        nbformat_minor: 5,
    };
}

/** The notebook as text for a download. */
export function trainingNotebookText(script, options) {
    return `${JSON.stringify(trainingNotebook(script, options), null, 1)}\n`;
}
