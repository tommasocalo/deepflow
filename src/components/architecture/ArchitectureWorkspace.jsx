import { useCallback, useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { ArchitectureStoreProvider, createArchitecturePaneStore, useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import ArchitectureCanvas from './ArchitectureCanvas.jsx';
import ArchitectureRibbon from './ArchitectureRibbon.jsx';
import LevelTabs from './LevelTabs.jsx';
import NodeInspector from './NodeInspector.jsx';
import RepeatIterationBar from './RepeatIterationBar.jsx';
import { OpenToSideProvider } from './OpenToSideContext.jsx';
import { useArchitectureHistoryKeys } from '../../hooks/useArchitectureHistoryKeys.js';
import './workspace.css';

/** Canvas-only capabilities remain inside their own ReactFlow provider. */
function CanvasCommandBridge({ paneId, register }) {
    const markerRef = useRef(null);
    const architectureStore = useArchitectureStoreApi();
    const { screenToFlowPosition, getNodes, fitView } = useReactFlow();

    useEffect(() => register(paneId, {
        fitView: () => fitView({ padding: 0.2, minZoom: 0.2, maxZoom: 1, duration: 220 }),
        insert: (entry, isTemplate) => {
            const bounds = markerRef.current?.closest('[data-architecture-canvas]')?.getBoundingClientRect();
            if (!bounds) return null;
            const center = screenToFlowPosition({ x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height * 0.4 });
            const topLeft = screenToFlowPosition({ x: bounds.left + 24, y: bounds.top + 55 });
            const bottomRight = screenToFlowPosition({ x: bounds.right - 24, y: bounds.bottom - 70 });
            const minX = topLeft.x + 120;
            const maxX = bottomRight.x - 120;
            const clamp = (position) => ({
                x: minX <= maxX ? Math.max(minX, Math.min(position.x, maxX)) : center.x,
                y: Math.max(topLeft.y, Math.min(position.y, bottomRight.y - 60)),
            });
            let position = clamp({ x: Math.round(center.x), y: Math.round(center.y) });
            const nodes = getNodes();
            for (let attempt = 0; attempt < 12; attempt++) {
                if (!nodes.some((node) => Math.abs(node.position.x - position.x) < 220 && Math.abs(node.position.y - position.y) < 100)) break;
                position = clamp({ x: position.x + 32, y: position.y + 42 });
            }
            const state = architectureStore.getState();
            const id = entry.standardComponentKey ? state.addStandardComponent(entry.standardComponentKey, position)
                : isTemplate ? state.addSuperNodeInstance(entry.key, position, entry.name) : state.addLayerByLibraryKey(entry.key, position);
            if (id) { state.selectOnly(id); state.clearExpanded(); }
            return id;
        },
    }), [paneId, register, architectureStore, screenToFlowPosition, getNodes, fitView]);

    return <span ref={markerRef} hidden />;
}

function GraphPane({ id, active, onActivate, onClose, registerCommands }) {
    const importSource = useArchitectureStore((state) => state.graph.importSource);
    return (
        <section className={`df-graph-pane${active ? ' is-active' : ''}`} data-graph-pane={id}
            aria-label={id === 'main' ? 'Main graph pane' : 'Side graph pane'}
            tabIndex={-1} onPointerDownCapture={onActivate} onFocusCapture={onActivate}
            onDropCapture={(event) => { onActivate(); event.currentTarget.focus({ preventScroll: true }); }}>
            <div className="df-graph-pane-header">
                <LevelTabs />
                {importSource?.provider === 'huggingface' && <span className="df-import-source" title={`Imported architecture from Hugging Face: ${importSource.repoId} at ${importSource.commit}. Weights were not loaded.`}>
                    <span>HF import</span><strong className="df-import-source-full">{importSource.repoId}</strong><strong className="df-import-source-short">{importSource.repoId.split('/').at(-1)}</strong>
                </span>}
                {onClose && <button type="button" className="df-side-close" onClick={onClose} aria-label="Close side view" title="Close side view"><X size={13} strokeWidth={3} /></button>}
            </div>
            <RepeatIterationBar />
            <div className="df-architecture-body">
                <ArchitectureCanvas paneId={id} keyboardActive={active} overlay={(
                    <>
                        <NodeInspector />
                        <CanvasCommandBridge paneId={id} register={registerCommands} />
                    </>
                )} />
            </div>
        </section>
    );
}

/** Two independently navigable editors over the same architecture. */
export default function ArchitectureWorkspace() {
    useArchitectureHistoryKeys();
    const mainStore = useArchitectureStoreApi();
    const [sideStore, setSideStore] = useState(null);
    const [activePane, setActivePane] = useState('main');
    const [ratio, setRatio] = useState(50);
    const [sideFocusRequest, setSideFocusRequest] = useState(0);
    const workspaceRef = useRef(null);
    const resizing = useRef(false);
    const commands = useRef({});
    const registerCommands = useCallback((id, value) => {
        commands.current[id] = value;
        return () => { delete commands.current[id]; };
    }, []);

    useEffect(() => () => sideStore?.dispose(), [sideStore]);
    useEffect(() => {
        if (sideStore) workspaceRef.current?.querySelector('[data-graph-pane="side"] [role="tab"][aria-selected="true"]')?.focus({ preventScroll: true });
    }, [sideStore, sideFocusRequest]);

    const openToSide = useCallback((path) => {
        const sourceStore = activePane === 'side' && sideStore ? sideStore : mainStore;
        const inspection = sourceStore.getState().inspectedIterations;
        const destination = sideStore ?? createArchitecturePaneStore();
        // Start with the same inspected instances, including outer repeats;
        // subsequent index changes remain local to each pane.
        for (let depth = 1; depth <= path.length; depth++) {
            const prefix = path.slice(0, depth);
            destination.getState().setInspectedIteration(prefix, inspection[JSON.stringify(prefix)] ?? 0);
        }
        destination.getState().openLevel(path);
        if (!sideStore) setSideStore(destination);
        setActivePane('side');
        setSideFocusRequest((value) => value + 1);
    }, [activePane, mainStore, sideStore]);

    const closeSide = () => {
        setSideStore(null);
        setActivePane('main');
    };
    const resize = (event) => {
        if (!resizing.current) return;
        const bounds = workspaceRef.current.getBoundingClientRect();
        setRatio(Math.max(30, Math.min(70, (event.clientX - bounds.left) / bounds.width * 100)));
    };

    return (
        <OpenToSideProvider value={openToSide}>
            <div className="df-architecture-editor">
            <ArchitectureStoreProvider store={activePane === 'side' && sideStore ? sideStore : mainStore}>
                <ArchitectureRibbon key={activePane} paneId={activePane} split={!!sideStore}
                    onInsert={(entry, isTemplate) => commands.current[activePane]?.insert(entry, isTemplate)}
                    onFitView={() => commands.current[activePane]?.fitView()} />
            </ArchitectureStoreProvider>
            <div ref={workspaceRef} className={`df-architecture-split${sideStore ? ' is-split' : ''}`}
                style={sideStore ? { gridTemplateColumns: `minmax(0, ${ratio}fr) 7px minmax(0, ${100 - ratio}fr)` } : undefined}>
                <GraphPane id="main" active={activePane === 'main'} onActivate={() => setActivePane('main')} registerCommands={registerCommands} />
                {sideStore && <>
                    <div className="df-pane-divider" role="separator" aria-label="Resize graph panes" aria-orientation="vertical"
                        aria-valuemin={30} aria-valuemax={70} aria-valuenow={Math.round(ratio)} tabIndex={0}
                        onPointerDown={(event) => { resizing.current = true; event.currentTarget.setPointerCapture(event.pointerId); event.preventDefault(); }}
                        onPointerMove={resize}
                        onPointerUp={(event) => { resizing.current = false; event.currentTarget.releasePointerCapture(event.pointerId); }}
                        onPointerCancel={() => { resizing.current = false; }}
                        onLostPointerCapture={() => { resizing.current = false; }}
                        onDoubleClick={() => setRatio(50)}
                        onKeyDown={(event) => {
                            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
                            event.preventDefault();
                            setRatio((value) => event.key === 'Home' ? 30 : event.key === 'End' ? 70 : Math.max(30, Math.min(70, value + (event.key === 'ArrowRight' ? 5 : -5))));
                        }} />
                    <ArchitectureStoreProvider store={sideStore}>
                        <GraphPane id="side" active={activePane === 'side'} onActivate={() => setActivePane('side')} onClose={closeSide} registerCommands={registerCommands} />
                    </ArchitectureStoreProvider>
                </>}
            </div>
            </div>
        </OpenToSideProvider>
    );
}
