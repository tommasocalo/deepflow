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
import SuperNodeInstanceView from './SuperNodeInstanceView.jsx';
import RepeatNodeView from './RepeatNodeView.jsx';
import GradientEdge from './GradientEdge.jsx';
import GraphStatusPanel from './GraphStatusPanel.jsx';

const nodeTypes = {
    archLayer: LayerNodeView,
    archSupernode: SuperNodeInstanceView,
    archRepeat: RepeatNodeView,
};

const edgeTypes = {
    archEdge: GradientEdge,
};

const VIEWPORT_KEY_PREFIX = 'deepflow:architecture-viewport';

function viewportKey(currentPath) {
    if (!currentPath || currentPath.length === 0) return `${VIEWPORT_KEY_PREFIX}:root`;
    return `${VIEWPORT_KEY_PREFIX}:${currentPath.join('/')}`;
}

function toFlowNodes(graph, selectedSet) {
    return Object.values(graph.nodes).map((node) => ({
        id: node.id,
        type:
            node.kind === 'supernode' ? 'archSupernode'
            : node.kind === 'repeat'  ? 'archRepeat'
            :                            'archLayer',
        position: node.position,
        data: { node },
        selected: selectedSet.has(node.id),
    }));
}

function toFlowEdges(graph, selectedEdgeSet, edgeMenuPositions, edgeMenuVersions) {
    return graph.edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
        type: 'archEdge',
        data: {
            gradientOp: e.gradientOp ?? 'none',
            gradientOpLambda: e.gradientOpLambda,
            menuPosition: edgeMenuPositions[e.id],
            menuVersion: edgeMenuVersions[e.id] ?? 0,
        },
        animated: false,
        selected: selectedEdgeSet.has(e.id),
        // Wider invisible hit area so users can actually click an edge.
        interactionWidth: 20,
    }));
}

const Inner = ({ children, overlay }) => {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const selectedNodeIds = useArchitectureStore((s) => s.selectedNodeIds);
    const templates = useTemplatesStore((s) => s.templates);
    const moveNode = useArchitectureStore((s) => s.moveNode);
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

    const { screenToFlowPosition, setViewport, fitView } = useReactFlow();
    const nodesInitialized = useNodesInitialized();
    const wrapperRef = useRef(null);
    const lastViewportKeyRef = useRef(null);
    const [interactionMode, setInteractionMode] = useState('select');  // 'select' | 'pan'
    const [selectedEdgeIds, setSelectedEdgeIds] = useState([]);
    const [edgeMenuPositions, setEdgeMenuPositions] = useState({});
    const [edgeMenuVersions, setEdgeMenuVersions] = useState({});
    const [spacePressed, setSpacePressed] = useState(false);

    // Resolve the visible graph from the path. Templates is included in deps
    // so edits inside a drilled-in template re-render the canvas.
    const viewGraph = useMemo(
        () => resolveGraphLevel(rootGraph, currentPath, templates).graph,
        [rootGraph, currentPath, templates],
    );

    const selectedSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds]);
    const selectedEdgeSet = useMemo(() => new Set(selectedEdgeIds), [selectedEdgeIds]);
    const nodes = useMemo(() => toFlowNodes(viewGraph, selectedSet), [viewGraph, selectedSet]);
    const edges = useMemo(
        () => toFlowEdges(viewGraph, selectedEdgeSet, edgeMenuPositions, edgeMenuVersions),
        [viewGraph, selectedEdgeSet, edgeMenuPositions, edgeMenuVersions],
    );
    const showPanCursor = interactionMode === 'pan' || spacePressed;
    const graphIsEmpty = nodes.length === 0;

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

    // Save and restore viewport per level so drilling in/out preserves zoom.
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

    const handleViewportChange = useCallback(
        (vp) => {
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
            for (const change of changes) {
                if (change.type === 'position' && change.position) {
                    moveNode(change.id, change.position);
                } else if (change.type === 'remove') {
                    removeNodeStore(change.id);
                } else if (change.type === 'select') {
                    setNodeSelected(change.id, change.selected);
                }
            }
        },
        [moveNode, removeNodeStore, setNodeSelected],
    );

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

    const onEdgeClick = useCallback(
        (event, edge) => {
            const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
            setSelectedEdgeIds([edge.id]);
            setEdgeMenuPositions((positions) => ({ ...positions, [edge.id]: position }));
            setEdgeMenuVersions((versions) => ({
                ...versions,
                [edge.id]: (versions[edge.id] ?? 0) + 1,
            }));
        },
        [screenToFlowPosition],
    );

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
                    nodes={nodes}
                    edges={edges}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onEdgeClick={onEdgeClick}
                    onConnect={onConnect}
                    onDragOver={onDragOver}
                    onDrop={onDrop}
                    onViewportChange={handleViewportChange}
                    onPaneClick={() => {
                        clearSelection();
                        clearExpanded();
                        setSelectedEdgeIds([]);
                        setEdgeMenuPositions({});
                        setEdgeMenuVersions({});
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
