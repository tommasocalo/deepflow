import { useEffect } from 'react';
import { Handle, Position, useUpdateNodeInternals } from '@xyflow/react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { getIncomingEdges, mapInputEdgesToPorts } from '../../domain/graphInputs.js';
import { activationLabel } from './nodeGeometry.js';
import { inputPortsFor } from './nodeInputPorts.js';

export default function NodeInputPorts({ node, shapes }) {
    const templates = useTemplatesStore((s) => s.templates);
    const getViewGraph = useArchitectureStore((s) => s.getViewGraph);
    const updateNodeInternals = useUpdateNodeInternals();
    const ports = inputPortsFor(node, templates);
    const portKey = ports.map((port) => port.id).join('|');
    useEffect(() => { updateNodeInternals(node.id); }, [node.id, node.mode, portKey, updateNodeInternals]);
    if (ports.length < 2) return <Handle type="target" position={Position.Top} className="df-handle" />;
    const isOperation = node.libraryKey?.startsWith('deepflow.op.');
    const edges = mapInputEdgesToPorts(getIncomingEdges(getViewGraph(), node.id), ports.map((port) => port.id), { broadcastSingle: !isOperation });
    return ports.map((port, index) => {
        const sidePort = isOperation && index === 1;
        const left = isOperation ? '50%' : `${(index + .5) / ports.length * 100}%`;
        const shape = activationLabel(shapes[edges[port.id]?.source]) ?? '—';
        return <span key={port.id}>
            <Handle id={port.id} type="target" position={sidePort ? Position.Right : Position.Top} className="df-handle" style={sidePort ? undefined : { left }} title={port.label} aria-label={`Input ${port.label}`} />
            <span className={`df-port-shape ${sidePort ? 'df-port-shape--right' : 'df-port-shape--input'}`} style={sidePort ? undefined : { left }} aria-label={`${port.label} input dimensions: ${shape}`}>
                {port.label}: {shape}
            </span>
        </span>;
    });
}
