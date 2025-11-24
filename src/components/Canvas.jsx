import React, { useCallback, useRef, useEffect } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    addEdge,
    useReactFlow,
    ReactFlowProvider
} from '@xyflow/react';

import CustomNode from './CustomNode';

const nodeTypes = {
    nnBlock: CustomNode,
};

const VIEWPORT_STORAGE_KEY = 'deepflow-viewport';

const CanvasContent = ({ nodes, edges, onNodesChange, onEdgesChange, setEdges, onAddNode }) => {
    const reactFlowWrapper = useRef(null);
    const { screenToFlowPosition, setViewport, getViewport, fitView } = useReactFlow();
    const initializedRef = useRef(false);

    // Load saved viewport on mount
    useEffect(() => {
        if (initializedRef.current) return;
        initializedRef.current = true;

        const savedViewport = localStorage.getItem(VIEWPORT_STORAGE_KEY);
        if (savedViewport) {
            try {
                const viewport = JSON.parse(savedViewport);
                // Small delay to ensure ReactFlow is ready
                setTimeout(() => {
                    setViewport(viewport, { duration: 0 });
                }, 0);
            } catch (error) {
                console.error('Failed to load viewport:', error);
                fitView();
            }
        } else {
            // Only fit view if no saved viewport
            fitView();
        }
    }, [setViewport, fitView]);

    // Save viewport on change
    const handleViewportChange = useCallback((viewport) => {
        localStorage.setItem(VIEWPORT_STORAGE_KEY, JSON.stringify(viewport));
    }, []);

    const onConnect = useCallback(
        (params) => setEdges((eds) => addEdge(params, eds)),
        [setEdges],
    );

    const onDragOver = useCallback((event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback(
        (event) => {
            event.preventDefault();

            const type = event.dataTransfer.getData('application/reactflow');

            // check if the dropped element is valid
            if (typeof type === 'undefined' || !type) {
                return;
            }

            const position = screenToFlowPosition({
                x: event.clientX,
                y: event.clientY,
            });

            onAddNode(type, position);
        },
        [screenToFlowPosition, onAddNode],
    );

    return (
        <div style={{ width: '100%', height: '100%', backgroundColor: '#ffffff' }} ref={reactFlowWrapper}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                onDragOver={onDragOver}
                onDrop={onDrop}
                onViewportChange={handleViewportChange}
                style={{ width: '100%', height: '100%' }}
            >
                <Background color="#000000" gap={24} size={1} variant="dots" style={{ opacity: 0.1 }} />
                <Controls className="!bg-surface !border-2 !border-black !shadow-hard !text-black [&>button]:!border-b-black [&>button]:!border-b-2 last:[&>button]:!border-b-0" />
            </ReactFlow>
        </div>
    );
};

const Canvas = (props) => (
    <ReactFlowProvider>
        <CanvasContent {...props} />
    </ReactFlowProvider>
);

export default Canvas;
