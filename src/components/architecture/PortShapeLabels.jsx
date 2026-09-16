import React from 'react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { describePorts } from './portShapes.js';

/** Persistent dimensions centered over the incoming and outgoing wires. */
export default function PortShapeLabels({ node, shapes }) {
    const getViewGraph = useArchitectureStore((state) => state.getViewGraph);
    const ports = describePorts(getViewGraph(), node, shapes);
    return <>
        {ports.input !== null && <span className="df-port-shape df-port-shape--input" aria-label={`Input dimensions: ${ports.input}`}>{ports.input}</span>}
        {ports.output !== null && <span className="df-port-shape df-port-shape--output" aria-label={`Output dimensions: ${ports.output}`}>{ports.output}</span>}
    </>;
}
