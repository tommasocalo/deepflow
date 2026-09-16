import React from 'react';
import { Boxes, Home, Repeat, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';

/**
 * Cards for every level currently open — `main` plus one card per
 * supernode entered — shown as a strip under the Architecture tab. Closing
 * a card returns to its parent level.
 */
export default function LevelTabs() {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const templates = useTemplatesStore((s) => s.templates);
    const goToLevel = useArchitectureStore((s) => s.goToLevel);

    const level = resolveGraphLevel(rootGraph, currentPath, templates);
    const cards = [{ id: 'root', label: 'main', depth: 0, kind: 'root' }];
    let graph = rootGraph;
    for (const seg of level.segments ?? []) {
        const node = graph?.nodes?.[seg.id];
        const tpl = node ? templates[node.templateId] : null;
        cards.push({
            id: seg.id,
            depth: seg.depth,
            label: seg.label,
            kind: node?.kind ?? 'supernode',
            count: node?.kind === 'repeat' ? node.count : null,
            template: tpl?.name,
        });
        graph = tpl?.graph;
    }
    const activeDepth = cards.length - 1;

    return (
        <div className="df-levels" role="tablist" aria-label="Open levels">
            {cards.map((card, i) => {
                const active = i === activeDepth;
                const Icon = card.kind === 'root' ? Home : card.kind === 'repeat' ? Repeat : Boxes;
                return (
                    <div key={`${card.depth}:${card.id}`} className={`df-level-card${active ? ' is-active' : ''}`}>
                        <button
                            type="button"
                            role="tab"
                            aria-selected={active}
                            className="df-level-tab"
                            onClick={() => goToLevel(card.depth)}
                            title={card.template ? `Template: ${card.template}` : 'Top level'}
                        >
                            <Icon size={12} strokeWidth={2.5} />
                            <span className="df-level-label">{card.label}</span>
                            {card.count != null && <span className="df-level-count">× {card.count}</span>}
                        </button>
                        {i > 0 && (
                            <button
                                type="button"
                                className="df-level-close"
                                onClick={() => goToLevel(card.depth - 1)}
                                aria-label={`Close ${card.label}`}
                                title="Close"
                            >
                                <X size={11} strokeWidth={3} />
                            </button>
                        )}
                    </div>
                );
            })}
            {cards.length > 1 && (
                <span className="df-levels-hint">Edits inside a template apply to every instance.</span>
            )}
        </div>
    );
}
