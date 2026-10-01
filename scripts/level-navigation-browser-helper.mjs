/** Persistent cards for the graph views opened in one pane. */
export const levelCards = (pane) => pane.getByRole('tablist', { name: 'Open levels', exact: true });
export const activeLevelCard = (pane) => levelCards(pane).getByRole('tab', { selected: true });

const exact = (label) => label instanceof RegExp ? label
    : new RegExp(`^${String(label).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`);

export function levelCard(pane, label) {
    return levelCards(pane).locator('.df-level-label').filter({ hasText: exact(label) }).locator('..');
}

export async function selectLevel(pane, label) {
    await levelCard(pane, label).click();
}
