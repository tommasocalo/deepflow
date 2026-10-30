import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    ControlButton,
    ReactFlowProvider,
    useReactFlow,
} from '@xyflow/react';
import { Hand, MousePointer2 } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore.js';
import OptimizerNodeView from './OptimizerNodeView.jsx';
import LossNodeView from './LossNodeView.jsx';
import PhaseNodeView from './PhaseNodeView.jsx';

const nodeTypes = {
    trainOptimizer: OptimizerNodeView,
    trainLoss: LossNodeView,
    trainPhase: PhaseNodeView,
};

const VIEWPORT_KEY = 'deepflow:training-viewport';
// Fit beside the TrainingPalette (240 px wide, top right) and the Controls (bottom left), never above 100 %. A narrow
// canvas has the palette folded to its header (TrainingPalette), so the graph fits below it.
const FIT_OPTIONS = { padding: { top: '24px', bottom: '24px', left: '56px', right: '40%' }, maxZoom: 1 };
const NARROW_FIT_OPTIONS = { padding: { top: '60px', bottom: '16px', left: '56px', right: '16px' }, maxZoom: 1 };
const fitOptionsFor = (element) => ((element?.getBoundingClientRect().width ?? Infinity) < 600 ? NARROW_FIT_OPTIONS : FIT_OPTIONS);

function toFlowNodes(graph, selectedSet) {
    return Object.values(graph.nodes).map((node) => ({
        id: node.id,
        type:
            node.kind === 'optimizer' ? 'trainOptimizer'
            : node.kind === 'loss'    ? 'trainLoss'
            : node.kind === 'phase'   ? 'trainPhase'
            :                            'default',
        position: node.position,
        data: { node },
        selected: selectedSet.has(node.id),
    }));
}

function toFlowEdges(graph, selectedEdgeSet) {
    return graph.edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: e.sourceHandle,
        targetHandle: e.targetHandle,
        animated: false,
        selected: selectedEdgeSet.has(e.id),
        style: { stroke: '#000000', strokeWidth: 2, strokeDasharray: '6 4' },
        interactionWidth: 20,
    }));
}

const Inner = ({ fitOnOpen = false }) => {
    const graph = useTrainingStore((s) => s.graph);
    const selectedNodeIds = useTrainingStore((s) => s.selectedNodeIds);
    const moveNode = useTrainingStore((s) => s.moveNode);
    const removeNode = useTrainingStore((s) => s.removeNode);
    const removeEdge = useTrainingStore((s) => s.removeEdge);
    const connect = useTrainingStore((s) => s.connect);
    const setNodeSelected = useTrainingStore((s) => s.setNodeSelected);
    const clearSelection = useTrainingStore((s) => s.clearSelection);
    const clearExpanded = useTrainingStore((s) => s.clearExpanded);
    const addOptimizer = useTrainingStore((s) => s.addOptimizer);
    const addLoss = useTrainingStore((s) => s.addLoss);
    const addPhase = useTrainingStore((s) => s.addPhase);

    const { screenToFlowPosition, setViewport, fitView } = useReactFlow();
    const wrapperRef = useRef(null);
    const initRef = useRef(false);
    const [interactionMode, setInteractionMode] = useState('select');
    const [selectedEdgeIds, setSelectedEdgeIds] = useState([]);
    const [spacePressed, setSpacePressed] = useState(false);

    const selectedSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds]);
    const selectedEdgeSet = useMemo(() => new Set(selectedEdgeIds), [selectedEdgeIds]);
    const nodes = useMemo(() => toFlowNodes(graph, selectedSet), [graph, selectedSet]);
    const edges = useMemo(() => toFlowEdges(graph, selectedEdgeSet), [graph, selectedEdgeSet]);
    const showPanCursor = interactionMode === 'pan' || spacePressed;

    useEffect(() => {
        const isEditable = (target) =>
            target?.closest?.('input, textarea, select, [contenteditable="true"]');
        const onKeyDown = (event) => {
            if (event.code === 'Space' && !isEditable(event.target)) setSpacePressed(true);
        };
        const onKeyUp = (event) => {
            if (event.code === 'Space') setSpacePressed(false);
        };
        window.addEventListener('keydown', onKeyDown);
        window.addEventListener('keyup', onKeyUp);
        window.addEventListener('blur', onKeyUp);
        return () => {
            window.removeEventListener('keydown', onKeyDown);
            window.removeEventListener('keyup', onKeyUp);
            window.removeEventListener('blur', onKeyUp);
        };
    }, []);

    useEffect(() => {
        if (initRef.current) return;
        initRef.current = true;
        // Edu lessons fit instead: a viewport saved in another workspace may not show the step's loss block.
        const saved = fitOnOpen ? null : localStorage.getItem(VIEWPORT_KEY);
        if (saved) {
            try { setViewport(JSON.parse(saved), { duration: 0 }); return; } catch {
                // A stale saved viewport falls back to fitting the graph below.
            }
        }
        setTimeout(() => fitView({ ...fitOptionsFor(wrapperRef.current), duration: 0 }), 0);
    }, [setViewport, fitView, fitOnOpen]);

    // Edu canvas bridge (docs/edu/SPEC.md §6.12): "Where is it?" asks for the same fit.
    useEffect(() => {
        const onCommand = ({ detail = {} }) => {
            if (detail.command !== 'fit') return;
            fitView({ ...fitOptionsFor(wrapperRef.current), duration: 220 });
            detail.result = true;
        };
        window.addEventListener('deepflow-edu:training-canvas', onCommand);
        return () => window.removeEventListener('deepflow-edu:training-canvas', onCommand);
    }, [fitView]);

    const handleViewportChange = useCallback((vp) => {
        try { localStorage.setItem(VIEWPORT_KEY, JSON.stringify(vp)); } catch {
            // Viewport persistence is optional when storage is unavailable.
        }
    }, []);

    const onNodesChange = useCallback(
        (changes) => {
            for (const change of changes) {
                if (change.type === 'position' && change.position) moveNode(change.id, change.position);
                else if (change.type === 'remove') removeNode(change.id);
                else if (change.type === 'select') setNodeSelected(change.id, change.selected);
            }
        },
        [moveNode, removeNode, setNodeSelected],
    );

    const onEdgesChange = useCallback(
        (changes) => {
            for (const change of changes) {
                if (change.type === 'remove') removeEdge(change.id);
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
        [removeEdge],
    );

    const onConnect = useCallback((params) => connect(params), [connect]);

    const onDragOver = useCallback((e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (e) => {
            e.preventDefault();
            const optType = e.dataTransfer.getData('application/deepflow-optimizer');
            const lossType = e.dataTransfer.getData('application/deepflow-loss');
            const phase = e.dataTransfer.getData('application/deepflow-phase');
            if (!optType && !lossType && !phase) return;
            const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
            if (optType) addOptimizer(optType, position);
            else if (lossType) addLoss(lossType, position);
            else if (phase) addPhase(position, { label: phase });
        },
        [screenToFlowPosition, addOptimizer, addLoss, addPhase],
    );

    return (
        <div
            ref={wrapperRef}
            className={showPanCursor ? 'deepflow-mode-pan' : 'deepflow-mode-select'}
            style={{ width: '100%', height: '100%', backgroundColor: '#ffffff' }}
        >
            <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onDragOver={onDragOver}
                onDrop={onDrop}
                onViewportChange={handleViewportChange}
                onPaneClick={() => { clearSelection(); clearExpanded(); setSelectedEdgeIds([]); }}
                selectionOnDrag={interactionMode === 'select'}
                panOnDrag={interactionMode === 'pan'}
                panActivationKeyCode="Space"
                multiSelectionKeyCode={['Meta', 'Shift']}
                deleteKeyCode={['Delete', 'Backspace']}
                edgesFocusable
                fitViewOptions={FIT_OPTIONS}
                proOptions={{ hideAttribution: true }}
                style={{ width: '100%', height: '100%' }}
            >
                <Background color="#f8d98a" gap={24} size={1.2} variant="dots" />
                <Controls className="df-flow-controls text-black!">
                    <ControlButton
                        onClick={() => setInteractionMode('select')}
                        title="Select nodes"
                        className={interactionMode === 'select' ? 'bg-black! text-white!' : 'bg-white! text-black!'}
                    >
                        <MousePointer2 size={12} strokeWidth={2.5} />
                    </ControlButton>
                    <ControlButton
                        onClick={() => setInteractionMode('pan')}
                        title="Pan canvas"
                        className={interactionMode === 'pan' ? 'bg-black! text-white!' : 'bg-white! text-black!'}
                    >
                        <Hand size={12} strokeWidth={2.5} />
                    </ControlButton>
                </Controls>
            </ReactFlow>
            {spacePressed && <div className="df-space-hint">Space: drag canvas</div>}
        </div>
    );
};

const TrainingCanvas = ({ fitOnOpen = false }) => (
    <ReactFlowProvider>
        <Inner fitOnOpen={fitOnOpen} />
    </ReactFlowProvider>
);

export default TrainingCanvas;
