import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    ControlButton,
    ConnectionLineType,
    ReactFlowProvider,
    useReactFlow,
    useStore as useFlowStore,
} from '@xyflow/react';
import { Hand, MousePointer2 } from 'lucide-react';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { isValidConnection, resolveGraphLevel } from '../../domain/architecture/index.js';
import LayerNodeView from './LayerNodeView.jsx';
import OpNodeView from './OpNodeView.jsx';
import SuperNodeInstanceView from './SuperNodeInstanceView.jsx';
import RepeatNodeView from './RepeatNodeView.jsx';
import ArchitectureEdge from './ArchitectureEdge.jsx';
import GraphStatusPanel from './GraphStatusPanel.jsx';
import GradientFlowControls from './GradientFlowControls.jsx';
import { useGradientFlow } from '../../hooks/useGradientFlow.js';
import { residualEdgeIds } from './residualEdges.js';
import { tensorFlowNodeIds } from './tensorFlow.js';
import { createDragBatcher, syncFlowNodes } from './flowAdapter.js';
import { isBlankCanvasDrop } from './connectionGesture.js';
import './architecture.css';
import './connectors.css';

const nodeTypes = {
    archLayer: LayerNodeView,
    archOp: OpNodeView,
    archSupernode: SuperNodeInstanceView,
    archRepeat: RepeatNodeView,
};

const edgeTypes = {
    archEdge: ArchitectureEdge,
};

const NODE_ORIGIN = [0.5, 0];

const VIEWPORT_KEY_PREFIX = 'deepflow:architecture-viewport';

function viewportKey(currentPath, paneId) {
    const prefix = paneId === 'main' ? VIEWPORT_KEY_PREFIX : `${VIEWPORT_KEY_PREFIX}:${paneId}`;
    if (!currentPath || currentPath.length === 0) return `${prefix}:root`;
    return `${prefix}:${currentPath.join('/')}`;
}

function toFlowEdges(edges, selectedEdgeSet, residuals, gradients) {
    return edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle: edge.sourceHandle,
        targetHandle: edge.targetHandle,
        type: 'archEdge',
        data: { detour: residuals.has(edge.id), gradient: gradients?.edges[edge.id] },
        selected: selectedEdgeSet.has(edge.id),
        animated: false,
        interactionWidth: 20,
        ariaLabel: 'Connection. Drag either endpoint to reconnect; release on the canvas to disconnect.',
    }));
}

const Inner = ({ children, overlay, paneId = 'main', keyboardActive = true }) => {
    const architectureStore = useArchitectureStoreApi();
    const storeId = useMemo(() => architectureStore.getState().getMutationContext().storeId, [architectureStore]);
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const selectedNodeIds = useArchitectureStore((s) => s.selectedNodeIds);
    const templates = useTemplatesStore((s) => s.templates);
    const moveNodes = useArchitectureStore((s) => s.moveNodes);
    const removeNodeStore = useArchitectureStore((s) => s.removeNode);
    const removeEdgeStore = useArchitectureStore((s) => s.removeEdge);
    const connect = useArchitectureStore((s) => s.connect);
    const reconnectEdge = useArchitectureStore((s) => s.reconnectEdge);
    const setNodeSelected = useArchitectureStore((s) => s.setNodeSelected);
    const clearSelection = useArchitectureStore((s) => s.clearSelection);
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const clearExpanded = useArchitectureStore((s) => s.clearExpanded);
    const addLayerByLibraryKey = useArchitectureStore((s) => s.addLayerByLibraryKey);
    const addSuperNodeInstance = useArchitectureStore((s) => s.addSuperNodeInstance);
    const addStandardComponent = useArchitectureStore((s) => s.addStandardComponent);
    const showGradientFlow = useArchitectureStore((s) => s.showGradientFlow);
    const gradientReport = useGradientFlow(showGradientFlow);

    const { screenToFlowPosition, getViewport, setViewport, setNodes, fitView } = useReactFlow();
    const wrapperRef = useRef(null);
    const lastViewportKeyRef = useRef(null);
    const mountedRef = useRef(false);
    const connectionContextRef = useRef(null);
    const reconnectRef = useRef(null);
    const [dragBatcher] = useState(createDragBatcher);
    const [interactionMode, setInteractionMode] = useState('select');  // 'select' | 'pan'
    const [selectedEdgeIds, setSelectedEdgeIds] = useState([]);
    const [spacePressed, setSpacePressed] = useState(false);

    // Resolve the visible graph from the path. Templates is included in deps
    // so edits inside a drilled-in template re-render the canvas.
    const viewGraph = useMemo(
        () => resolveGraphLevel(rootGraph, currentPath, templates).graph,
        [rootGraph, currentPath, templates],
    );

    const levelKey = JSON.stringify(currentPath);
    const dragContext = useMemo(() => ({ graph: viewGraph, levelKey, storeId }), [viewGraph, levelKey, storeId]);
    const tensorFlowIds = useMemo(() => tensorFlowNodeIds(viewGraph), [viewGraph]);
    const [initialNodes] = useState(() => syncFlowNodes([], viewGraph, selectedNodeIds, tensorFlowIds, levelKey));
    const selectedEdgeSet = useMemo(() => new Set(selectedEdgeIds), [selectedEdgeIds]);
    // Moving nodes does not change the topology or rebuild stationary wires.
    const residuals = useMemo(() => residualEdgeIds(viewGraph.edges), [viewGraph.edges]);
    const edges = useMemo(
        () => toFlowEdges(viewGraph.edges, selectedEdgeSet, residuals, gradientReport),
        [viewGraph.edges, selectedEdgeSet, residuals, gradientReport],
    );
    const showPanCursor = interactionMode === 'pan' || (keyboardActive && spacePressed);
    const nodeCount = Object.keys(viewGraph.nodes).length;
    const graphIsEmpty = nodeCount === 0;
    // One boolean subscription also changes when navigating between instances
    // of the same template, where IDs and measured dimensions are reused.
    const levelNodesReady = useFlowStore(useCallback((state) => state.nodes.length === nodeCount
        && state.nodes.every((node) => node.data.levelKey === levelKey && node.measured?.width > 0 && node.measured?.height > 0),
    [nodeCount, levelKey]));

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            dragBatcher.cancel();
            reconnectRef.current = null;
            connectionContextRef.current = null;
        };
    }, [dragBatcher]);

    useEffect(() => {
        const resetPositions = dragBatcher.reconcile(dragContext);
        setNodes((previous) => syncFlowNodes(previous, viewGraph, selectedNodeIds, tensorFlowIds, levelKey, resetPositions));
    }, [setNodes, dragBatcher, dragContext, viewGraph, selectedNodeIds, tensorFlowIds, levelKey]);

    const restorePositions = useCallback(() => {
        setNodes((previous) => syncFlowNodes(previous, viewGraph, selectedNodeIds, tensorFlowIds, levelKey, true));
    }, [setNodes, viewGraph, selectedNodeIds, tensorFlowIds, levelKey]);

    useEffect(() => {
        const cancelDrag = () => {
            reconnectRef.current = null;
            connectionContextRef.current = null;
            dragBatcher.cancel();
            restorePositions();
        };
        window.addEventListener('pointercancel', cancelDrag);
        window.addEventListener('blur', cancelDrag);
        return () => {
            window.removeEventListener('pointercancel', cancelDrag);
            window.removeEventListener('blur', cancelDrag);
        };
    }, [dragBatcher, restorePositions]);

    useEffect(() => {
        const isEditable = (target) =>
            target?.closest?.('input, textarea, select, [contenteditable="true"]');
        const onKeyDown = (event) => {
            if (!keyboardActive) return;
            if (event.code === 'Space' && !isEditable(event.target)) setSpacePressed(true);
            if (event.key === 'Escape') {
                reconnectRef.current = null;
                connectionContextRef.current = null;
            }
        };
        const onKeyUp = (event) => {
            if (event.code === 'Space') setSpacePressed(false);
        };
        const onBlur = () => setSpacePressed(false);
        window.addEventListener('keydown', onKeyDown);
        window.addEventListener('keyup', onKeyUp);
        window.addEventListener('blur', onBlur);
        return () => {
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup', onKeyUp);
            window.removeEventListener('blur', onBlur);
        };
    }, [keyboardActive]);

    // Keep the same graph point centred as a split opens, closes or resizes.
    // A new level starts a new baseline: its iteration bar may change canvas
    // height, but its saved viewport must not receive an extra resize offset.
    useEffect(() => {
        const canvas = wrapperRef.current;
        let previous = canvas.getBoundingClientRect();
        const observer = new ResizeObserver(() => {
            const next = canvas.getBoundingClientRect();
            if (previous.width > 0 && previous.height > 0 && next.width > 0 && next.height > 0
                && (next.width !== previous.width || next.height !== previous.height)) {
                const viewport = getViewport();
                setViewport({ ...viewport, x: viewport.x + (next.width - previous.width) / 2, y: viewport.y + (next.height - previous.height) / 2 }, { duration: 0 });
            }
            previous = next;
        });
        observer.observe(canvas);
        return () => observer.disconnect();
    }, [getViewport, setViewport, levelKey]);

    // Restore viewport per level; movement is saved once in onMoveEnd.
    useEffect(() => {
        // setNodes is queued; initialization can still describe the previous
        // level until the destination's own measured nodes reach ReactFlow.
        if (!levelNodesReady) return;
        const key = viewportKey(currentPath, paneId);
        if (lastViewportKeyRef.current === key) return;
        lastViewportKeyRef.current = key;

        try {
            const saved = JSON.parse(localStorage.getItem(key));
            if (saved && [saved.x, saved.y, saved.zoom].every(Number.isFinite) && saved.zoom > 0) {
                setViewport({ ...saved, zoom: Math.max(0.2, Math.min(2, saved.zoom)) }, { duration: 0 });
                return;
            }
        } catch {
            // A stale viewport or disabled storage should not prevent editing.
        }
        if (graphIsEmpty) {
            setViewport({ x: 0, y: 0, zoom: 1 }, { duration: 0 });
        } else {
            fitView({ padding: 0.2, minZoom: 0.2, maxZoom: 1, duration: 0 });
        }
    }, [currentPath, paneId, levelNodesReady, graphIsEmpty, setViewport, fitView]);

    const handleMoveEnd = useCallback(
        (_event, vp) => {
            const key = viewportKey(currentPath, paneId);
            try {
                localStorage.setItem(key, JSON.stringify(vp));
            } catch {
                // Viewport persistence is optional when storage is unavailable.
            }
        },
        [currentPath, paneId],
    );

    const onNodesChange = useCallback(
        (changes) => {
            if (!mountedRef.current) return;
            const transaction = dragBatcher.consume(changes, dragContext);
            if (transaction) {
                const expected = transaction.isDrag ? transaction.context : { levelKey, storeId };
                if (transaction.cancelled || !moveNodes(transaction.moves, expected)) restorePositions();
            }
            for (const change of changes) {
                if (change.type === 'remove' && viewGraph.nodes[change.id]) {
                    removeNodeStore(change.id);
                } else if (change.type === 'select') {
                    setNodeSelected(change.id, change.selected);
                }
            }
        },
        [dragBatcher, dragContext, levelKey, storeId, viewGraph, moveNodes, restorePositions, removeNodeStore, setNodeSelected],
    );

    const onDragStart = useCallback((_event, nodeOrNodes) => {
        // Dismiss on actual movement, leaving a simple click free to inspect.
        if (architectureStore.getState().expandedNodeId) clearExpanded();
        dragBatcher.begin(dragContext);
        const dragged = Array.isArray(nodeOrNodes) ? nodeOrNodes : [nodeOrNodes];
        if (dragged.some((node) => node?.data.levelKey !== levelKey)) dragBatcher.cancel();
    }, [architectureStore, clearExpanded, dragBatcher, dragContext, levelKey]);

    const onEdgesChange = useCallback(
        (changes) => {
            for (const change of changes) {
                if (change.type === 'remove') removeEdgeStore(change.id, { levelKey });
                else if (change.type === 'select') {
                    setSelectedEdgeIds((ids) => {
                        const next = new Set(ids);
                        if (change.selected) next.add(change.id);
                        else next.delete(change.id);
                        return [...next];
                    });
                }
            }
        },
        [removeEdgeStore, levelKey],
    );

    const onConnectStart = useCallback(() => {
        connectionContextRef.current = dragContext;
    }, [dragContext]);

    const onConnect = useCallback((connection) => {
        const context = connectionContextRef.current;
        if (context) connect(connection, context);
    }, [connect]);

    const onReconnectStart = useCallback((_event, edge) => {
        reconnectRef.current = { id: edge.id, context: dragContext, attempted: false };
    }, [dragContext]);

    const onReconnect = useCallback((edge, connection) => {
        const gesture = reconnectRef.current;
        if (!gesture || gesture.id !== edge.id) return;
        // An attempted drop must never subsequently remove the old connection,
        // even when a concurrent graph edit makes the commit stale.
        gesture.attempted = true;
        reconnectEdge(edge.id, connection, gesture.context);
    }, [reconnectEdge]);

    const onReconnectEnd = useCallback((event, edge, _handleType, connectionState) => {
        const gesture = reconnectRef.current;
        reconnectRef.current = null;
        if (!gesture || gesture.id !== edge.id || gesture.attempted) return;
        if (isBlankCanvasDrop(event, connectionState, wrapperRef.current)) {
            removeEdgeStore(edge.id, gesture.context);
        }
    }, [removeEdgeStore]);

    const validateConnection = useCallback((connection) => {
        const graph = architectureStore.getState().getViewGraph();
        return isValidConnection(graph, connection, reconnectRef.current?.id);
    }, [architectureStore]);

    const onDragOver = useCallback((e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    }, []);

    const onDrop = useCallback(
        (e) => {
            e.preventDefault();
            const libraryKey = e.dataTransfer.getData('application/deepflow-lib-key');
            const templateId = e.dataTransfer.getData('application/deepflow-template-id');
            const componentKey = e.dataTransfer.getData('application/deepflow-component-key');
            if (!libraryKey && !templateId && !componentKey) return;
            const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
            const id = componentKey ? addStandardComponent(componentKey, position) : templateId
                ? addSuperNodeInstance(templateId, position, templates[templateId]?.name)
                : addLayerByLibraryKey(libraryKey, position);
            if (id) {
                selectOnly(id);
                clearExpanded();
            }
        },
        [screenToFlowPosition, addLayerByLibraryKey, addSuperNodeInstance, addStandardComponent, templates, selectOnly, clearExpanded],
    );

    return (
        <div className="df-architecture-workspace" style={{ display: 'flex', width: '100%', height: '100%', minWidth: 0 }}>
            {children}
            <div
                ref={wrapperRef}
                data-architecture-canvas
                className={showPanCursor ? 'deepflow-mode-pan' : 'deepflow-mode-select'}
                style={{ flex: 1, minWidth: 0, height: '100%', backgroundColor: '#fff', position: 'relative' }}
            >
                <GraphStatusPanel />
                <ReactFlow
                    id={`architecture-${paneId}`}
                    defaultNodes={initialNodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    nodeOrigin={NODE_ORIGIN}
                    onNodesChange={onNodesChange}
                    onNodeDragStart={onDragStart}
                    onSelectionDragStart={onDragStart}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    onConnectStart={onConnectStart}
                    onConnectEnd={() => { connectionContextRef.current = null; }}
                    isValidConnection={validateConnection}
                    onReconnectStart={onReconnectStart}
                    onReconnect={onReconnect}
                    onReconnectEnd={onReconnectEnd}
                    connectionLineType={ConnectionLineType.Bezier}
                    connectionLineStyle={{ stroke: '#111', strokeWidth: 2 }}
                    connectionDragThreshold={4}
                    reconnectRadius={12}
                    onDragOver={onDragOver}
                    onDrop={onDrop}
                    onMoveEnd={handleMoveEnd}
                    onPaneClick={() => {
                        clearSelection();
                        clearExpanded();
                        setSelectedEdgeIds([]);
                    }}
                    selectionOnDrag={interactionMode === 'select'}
                    panOnDrag={interactionMode === 'pan'}
                    panActivationKeyCode={keyboardActive ? 'Space' : null}
                    multiSelectionKeyCode={['Meta', 'Shift']}
                    deleteKeyCode={keyboardActive ? ['Delete', 'Backspace'] : null}
                    edgesFocusable
                    edgesReconnectable
                    minZoom={0.2}
                    maxZoom={2}
                    fitViewOptions={{ padding: 0.2, minZoom: 0.2, maxZoom: 1 }}
                    proOptions={{ hideAttribution: true }}
                    style={{ width: '100%', height: '100%' }}
                >
                    <Background color="#cbd5e1" gap={24} size={1} variant="dots" />
                    <Controls className="df-flow-controls text-black!" fitViewOptions={{ padding: 0.2, minZoom: 0.2, maxZoom: 1 }}>
                        <ControlButton
                            onClick={() => setInteractionMode('select')}
                            title="Select nodes"
                            aria-label="Select nodes"
                            className={interactionMode === 'select' ? 'bg-black! text-white!' : 'bg-white! text-black!'}
                        >
                            <MousePointer2 size={12} strokeWidth={2.5} />
                        </ControlButton>
                        <ControlButton
                            onClick={() => setInteractionMode('pan')}
                            title="Pan canvas"
                            aria-label="Pan canvas"
                            className={interactionMode === 'pan' ? 'bg-black! text-white!' : 'bg-white! text-black!'}
                        >
                            <Hand size={12} strokeWidth={2.5} />
                        </ControlButton>
                    </Controls>
                </ReactFlow>
                {overlay}
                {showGradientFlow && <GradientFlowControls graph={viewGraph} report={gradientReport} edge={selectedEdgeIds.length === 1 ? viewGraph.edges.find((edge) => edge.id === selectedEdgeIds[0]) : null} onClose={() => setSelectedEdgeIds([])} />}
                {keyboardActive && spacePressed && <div className="df-space-hint">Space: drag canvas</div>}
            </div>
        </div>
    );
};

const ArchitectureCanvas = ({ children, overlay, ...props }) => (
    <ReactFlowProvider>
        <Inner overlay={overlay} {...props}>{children}</Inner>
    </ReactFlowProvider>
);

export default ArchitectureCanvas;
