import { create } from 'zustand';
import { eduKey, readJSON, writeJSON } from '../lessons/engine/storage.js';

/**
 * Edu shell UI state (docs/edu/SPEC.md §6.12): the active top-level tab and the code panel. Only `activeTab` is
 * persisted. The lesson session listens to tab changes to record visited tabs.
 */
export const TABS = Object.freeze(['learn', 'architecture', 'data', 'training']);
export const UI_KEY = eduKey('ui:v1');
const CODE_TABS = ['model', 'train'];

function savedTab() {
    const tab = readJSON(UI_KEY, null)?.activeTab;
    return TABS.includes(tab) ? tab : 'learn';
}

export const useEduUiStore = create((set, get) => ({
    activeTab: savedTab(),
    codeOpen: false,
    codeTab: 'model',
    /** Collapsed state of the lesson drawer on narrow screens. */
    drawerOpen: true,
    setTab: (tab) => {
        if (!TABS.includes(tab) || get().activeTab === tab) return;
        // The code panel belongs to the model; it never floats over the Learn home.
        set(tab === 'learn' ? { activeTab: tab, codeOpen: false } : { activeTab: tab });
    },
    openCode: (tab) => set({ codeOpen: true, codeTab: CODE_TABS.includes(tab) ? tab : get().codeTab }),
    closeCode: () => set({ codeOpen: false }),
    setCodeTab: (tab) => { if (CODE_TABS.includes(tab)) set({ codeTab: tab }); },
    setDrawerOpen: (open) => set({ drawerOpen: !!open }),
}));

useEduUiStore.subscribe((state, previous) => {
    if (state.activeTab !== previous.activeTab) writeJSON(UI_KEY, { activeTab: state.activeTab });
});
