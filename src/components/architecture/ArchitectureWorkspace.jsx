import { useCallback, useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { ArchitectureStoreProvider, createArchitecturePaneStore } from '../../stores/architectureStore.js';
import ArchitectureCanvas from './ArchitectureCanvas.jsx';
import BlockPalette from './BlockPalette.jsx';
import LevelTabs from './LevelTabs.jsx';
import NodeInspector from './NodeInspector.jsx';
import SelectionToolbar from './SelectionToolbar.jsx';
import { OpenToSideProvider } from './OpenToSideContext.jsx';
import './workspace.css';

function GraphPane({ id, active, split, onActivate, onClose }) {
    return (
        <section className={`df-graph-pane${active ? ' is-active' : ''}`} data-graph-pane={id}
            aria-label={id === 'main' ? 'Main graph pane' : 'Side graph pane'}
            tabIndex={-1} onPointerDownCapture={onActivate} onFocusCapture={onActivate}
            onDropCapture={(event) => { onActivate(); event.currentTarget.focus({ preventScroll: true }); }}>
            <div className="df-graph-pane-header">
                <LevelTabs />
                {onClose && <button type="button" className="df-side-close" onClick={onClose} aria-label="Close side view" title="Close side view"><X size={13} strokeWidth={3} /></button>}
            </div>
            <div className="df-architecture-body">
                <ArchitectureCanvas paneId={id} keyboardActive={active} overlay={(
                    <>
                        <NodeInspector />
                        <BlockPalette key={split ? 'compact' : 'full'} initiallyCollapsed={split} keyboardActive={active} />
                        <SelectionToolbar />
                    </>
                )} />
            </div>
        </section>
    );
}

/** Two independently navigable editors over the same architecture. */
export default function ArchitectureWorkspace() {
    const [sideStore, setSideStore] = useState(null);
    const [activePane, setActivePane] = useState('main');
    const [ratio, setRatio] = useState(50);
    const [sideFocusRequest, setSideFocusRequest] = useState(0);
    const workspaceRef = useRef(null);
    const resizing = useRef(false);

    useEffect(() => () => sideStore?.dispose(), [sideStore]);
    useEffect(() => {
        if (sideStore) workspaceRef.current?.querySelector('[data-graph-pane="side"] [role="tab"][aria-selected="true"]')?.focus({ preventScroll: true });
    }, [sideStore, sideFocusRequest]);

    const openToSide = useCallback((path) => {
        if (sideStore) sideStore.getState().openLevel(path);
        else setSideStore(createArchitecturePaneStore({ path }));
        setActivePane('side');
        setSideFocusRequest((value) => value + 1);
    }, [sideStore]);

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
            <div ref={workspaceRef} className={`df-architecture-split${sideStore ? ' is-split' : ''}`}
                style={sideStore ? { gridTemplateColumns: `minmax(0, ${ratio}fr) 7px minmax(0, ${100 - ratio}fr)` } : undefined}>
                <GraphPane id="main" active={activePane === 'main'} split={!!sideStore} onActivate={() => setActivePane('main')} />
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
                        <GraphPane id="side" active={activePane === 'side'} split onActivate={() => setActivePane('side')} onClose={closeSide} />
                    </ArchitectureStoreProvider>
                </>}
            </div>
        </OpenToSideProvider>
    );
}
