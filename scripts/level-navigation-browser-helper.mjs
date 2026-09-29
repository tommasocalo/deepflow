/** Locate the compact level switcher in a graph pane. */
export const levelTrigger = (pane) => pane.locator('[data-level-trigger]');

export async function openLevelMenu(pane) {
    const trigger = levelTrigger(pane);
    if (await trigger.getAttribute('aria-expanded') !== 'true') await trigger.click();
    return pane.getByRole('dialog', { name: 'Open levels', exact: true });
}

const exact = (label) => label instanceof RegExp ? label
    : new RegExp(`^${String(label).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`);

export function levelEntry(menu, label) {
    return menu.locator('.df-level-entry-text > strong').filter({ hasText: exact(label) }).locator('xpath=../..');
}

export async function selectLevel(pane, label) {
    const menu = await openLevelMenu(pane);
    await levelEntry(menu, label).click();
}
