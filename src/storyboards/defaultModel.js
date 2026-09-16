const SEEDED_KEY = 'deepflow:seeded';
const WORKSPACE_KEYS = [
    'deepflow:architecture-graph',
    'deepflow:training-graph',
    'deepflow:templates',
];

/**
 * Initialize a default only when storage confirms there is no saved workspace.
 * The stores currently hydrate synchronously from localStorage on import.
 * Check the saved records themselves: an empty graph is still a user project,
 * and malformed records must be preserved rather than replaced with a demo.
 */
export function createDefaultModelSeeder({ seed, getStorage = () => window.localStorage }) {
    let attempted = false;

    return () => {
        // React StrictMode can run the mounting effect twice. Also avoid retrying
        // a partially interrupted initialization during the same app session.
        if (attempted) return false;
        attempted = true;

        try {
            const storage = getStorage();
            if ([SEEDED_KEY, ...WORKSPACE_KEYS].some((key) => storage.getItem(key) !== null)) {
                return false;
            }
            // Verify writes are possible before the storyboard mutates stores.
            // This also prevents reseeding after remounting or refreshing.
            storage.setItem(SEEDED_KEY, '1');
        } catch {
            // If storage cannot be inspected or written, leave all stores alone.
            return false;
        }

        try {
            seed();
            return true;
        } catch (error) {
            // A quota/storage failure can still occur during the subsequent
            // store writes. Keep the app usable and do not reset it again.
            console.warn('[DeepFlow] Default model initialization did not complete.', error);
            return false;
        }
    };
}
