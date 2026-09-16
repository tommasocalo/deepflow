import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    ControlButton,
    ReactFlowProvider,
    useNodesInitialized,
    useReactFlow,
} from '@xyflow/react';
import { Hand, MousePointer2 } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import LayerNodeView from './LayerNodeView.jsx';
import OpNodeView from './OpNodeView.jsx';
import SuperNodeInstanceView from './SuperNodeInstanceView.jsx';
import RepeatNodeView from './RepeatNodeView.jsx';
import ArchitectureEdge from './ArchitectureEdge.jsx';
import GraphStatusPanel from './GraphStatusPanel.jsx';
import { useValidation } from '../../hooks/useValidation.js';
import { COLUMN_HALF_WIDTH, activationLabel } from './nodeGeometry.js';
import { tensorFlowNodeIds } from './tensorFlow.js';
import { createDragBatcher, syncFlowNodes } from './flowAdapter.js';
import './architecture.css';

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

function viewportKey(currentPath) {
    if (!currentPath || currentPath.length === 0) return `${VIEWPORT_KEY_PREFIX}:root`;
    return `${VIEWPORT_KEY_PREFIX}:${currentPath.join('/')}`;
}

/**
 * An edge "skips" when another node sits vertically between its endpoints
 * in the same column — a residual / skip connection. Those are routed
 * around the column so they stay legible.
 */
function edgeSkips(edge, nodes) {
    const src = nodes[edge.source];
    const tgt = nodes[edge.target];
    if (!src || !tgt) return false;
    const top = Math.min(src.position.y, tgt.position.y) + 20;
    const bottom = Math.max(src.position.y, tgt.position.y) - 20;
    if (bottom - top < 40) return false;
    const cx = (src.position.x + tgt.position.x) / 2;
    return Object.values(nodes).some((n) =>
        n.id !== src.id && n.id !== tgt.id &&
        n.position.y > top && n.position.y < bottom &&
        Math.abs(n.position.x - cx) < COLUMN_HALF_WIDTH,
    );
}

function toFlowEdges(graph, selectedEdgeSet, shapes, tensorFlowIds) {
    return graph.edges.map((e) => {
        const detour = edgeSkips(e, graph.nodes);
        return {
            id: e.id,
            source: e.source,
            target: e.target,
            sourceHandle: e.sourceHandle,
            targetHandle: e.targetHandle,
            type: 'archEdge',
            data: {
                activation: activationLabel(shapes?.[e.source]),
                shape: shapes?.[e.source],
                tensorFlow: tensorFlowIds.has(e.source) && !detour,
                detour,
            },
            animated: false,
            selected: selectedEdgeSet.has(e.id),
            // Wider invisible hit area so users can actually click an edge.
            interactionWidth: 20,
        };
    });
}

const Inner = ({ children, overlay }) => {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const selectedNodeIds = useArchitectureStore((s) => s.selectedNodeIds);
    const templates = useTemplatesStore((s) => s.templates);
    const moveNodes = useArchitectureStore((s) => s.moveNodes);
    const removeNodeStore = useArchitectureStore((s) => s.removeNode);
    const removeEdgeStore = useArchitectureStore((s) => s.removeEdge);
    const connect = useArchitectureStore((s) => s.connect);
    const setNodeSelected = useArchitectureStore((s) => s.setNodeSelected);
    const clearSelection = useArchitectureStore((s) => s.clearSelection);
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const setExpanded = useArchitectureStore((s) => s.setExpanded);
    const clearExpanded = useArchitectureStore((s) => s.clearExpanded);
    const addLayerByLibraryKey = useArchitectureStore((s) => s.addLayerByLibraryKey);
    const addSuperNodeInstance = useArchitectureStore((s) => s.addSuperNodeInstance);

    const { screenToFlowPosition, setViewport, setNodes, fitView } = useReactFlow();
    const nodesInitialized = useNodesInitialized();
    const wrapperRef = useRef(null);
    const lastViewportKeyRef = useRef(null);
    const mountedRef = useRef(false);
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

    const validation = useValidation();
    const levelKey = JSON.stringify(currentPath);
    const dragContext = useMemo(() => ({ graph: viewGraph, levelKey }), [viewGraph, levelKey]);
    const tensorFlowIds = useMemo(() => tensorFlowNodeIds(viewGraph), [viewGraph]);
    const [initialNodes] = useState(() => syncFlowNodes([], viewGraph, selectedNodeIds, tensorFlowIds, levelKey));
    const selectedEdgeSet = useMemo(() => new Set(selectedEdgeIds), [selectedEdgeIds]);
    // Edges use committed geometry: their routing classification stays stable
    // while ReactFlow moves only the endpoint coordinates during a drag.
    const edges = useMemo(
        () => toFlowEdges(viewGraph, selectedEdgeSet, validation.shapes, tensorFlowIds),
        [viewGraph, selectedEdgeSet, validation.shapes, tensorFlowIds],
    );
    const showPanCursor = interactionMode === 'pan' || spacePressed;
    const graphIsEmpty = Object.keys(viewGraph.nodes).length === 0;

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            dragBatcher.cancel();
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
            if (event.code === 'Space' && !isEditable(event.target)) setSpacePressed(true);
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
    }, []);

    // Restore viewport per level; movement is saved once in onMoveEnd.
    useEffect(() => {
        if (!nodesInitialized && !graphIsEmpty) return;
        const key = viewportKey(currentPath);
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
    }, [currentPath, nodesInitialized, graphIsEmpty, setViewport, fitView]);

    const handleMoveEnd = useCallback(
        (_event, vp) => {
            const key = viewportKey(currentPath);
            try {
                localStorage.setItem(key, JSON.stringify(vp));
            } catch {
                // Viewport persistence is optional when storage is unavailable.
            }
        },
        [currentPath],
    );

    const onNodesChange = useCallback(
        (changes) => {
            if (!mountedRef.current) return;
            const transaction = dragBatcher.consume(changes, dragContext);
            if (transaction) {
                const expected = transaction.isDrag ? transaction.context : { levelKey };
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
        [dragBatcher, dragContext, levelKey, viewGraph, moveNodes, restorePositions, removeNodeStore, setNodeSelected],
    );

    const onDragStart = useCallback((_event, nodeOrNodes) => {
        dragBatcher.begin(dragContext);
        const dragged = Array.isArray(nodeOrNodes) ? nodeOrNodes : [nodeOrNodes];
        if (dragged.some((node) => node?.data.levelKey !== levelKey)) dragBatcher.cancel();
    }, [dragBatcher, dragContext, levelKey]);

    const onEdgesChange = useCallback(
        (changes) => {
            for (const change of changes) {
                if (change.type === 'remove') removeEdgeStore(change.id);
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
        [removeEdgeStore],
    );

    const onConnect = useCallback((params) => connect(params), [connect]);

    const onDragOver = useCallback((e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    }, []);

    const onDrop = useCallback(
        (e) => {
            e.preventDefault();
            const libraryKey = e.dataTransfer.getData('application/deepflow-lib-key');
            const templateId = e.dataTransfer.getData('application/deepflow-template-id');
            if (!libraryKey && !templateId) return;
            const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
            const id = templateId
                ? addSuperNodeInstance(templateId, position, templates[templateId]?.name)
                : addLayerByLibraryKey(libraryKey, position);
            if (id) {
                selectOnly(id);
                setExpanded(id);
            }
        },
        [screenToFlowPosition, addLayerByLibraryKey, addSuperNodeInstance, templates, selectOnly, setExpanded],
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
                    panActivationKeyCode="Space"
                    multiSelectionKeyCode={['Meta', 'Shift']}
                    deleteKeyCode={['Delete', 'Backspace']}
                    edgesFocusable
                    edgesReconnectable={false}
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
                {spacePressed && <div className="df-space-hint">Space: drag canvas</div>}
            </div>
        </div>
    );
};

const ArchitectureCanvas = ({ children, overlay }) => (
    <ReactFlowProvider>
        <Inner overlay={overlay}>{children}</Inner>
    </ReactFlowProvider>
);

export default ArchitectureCanvas;
