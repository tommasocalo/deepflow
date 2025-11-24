import React, { useCallback } from 'react';
import { useNodesState, useEdgesState } from '@xyflow/react';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import RightSidebar from './RightSidebar';
import Canvas from './Canvas';

const initialNodes = [
    {
        id: '1',
        type: 'nnBlock',
        position: { x: 250, y: 50 },
        data: { type: 'Input' }
    },
    {
        id: '2',
        type: 'nnBlock',
        position: { x: 250, y: 200 },
        data: { type: 'Dense', neurons: 128, activation: 'ReLU' }
    },
    {
        id: '3',
        type: 'nnBlock',
        position: { x: 250, y: 350 },
        data: { type: 'Output' }
    },
];

const initialEdges = [];

const Layout = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    const handleDeleteNode = useCallback((id) => {
        setNodes((nds) => nds.filter((node) => node.id !== id));
    }, [setNodes]);

    // Inject onDelete handler into node data
    const nodesWithHandlers = nodes.map(node => ({
        ...node,
        data: { ...node.data, onDelete: handleDeleteNode }
    }));

    const handleAddNode = useCallback((type, position) => {
        const newNode = {
            id: `${Date.now()}`,
            type: 'nnBlock',
            position,
            data: {
                type: type,
                neurons: 64,
                activation: 'ReLU',
                onDelete: handleDeleteNode
            },
        };
        setNodes((nds) => nds.concat(newNode));
    }, [setNodes, handleDeleteNode]);

    return (
        <div style={{ position: 'relative', width: '100%', height: '100%', backgroundColor: 'var(--color-bg)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {/* Navbar */}
            <Navbar />

            {/* Canvas and Sidebars Container */}
            <div style={{ position: 'relative', flex: 1, overflow: 'hidden' }}>
                {/* Canvas takes full space */}
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 0 }}>
                    <Canvas
                        nodes={nodesWithHandlers}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        setEdges={setEdges}
                        onDeleteNode={handleDeleteNode}
                        onAddNode={handleAddNode}
                    />
                </div>

                {/* Left Sidebar */}
                <Sidebar />

                {/* Right Sidebar (Block Palette) */}
                <RightSidebar onAddNode={handleAddNode} />
            </div>
        </div>
    );
};

export default Layout;
