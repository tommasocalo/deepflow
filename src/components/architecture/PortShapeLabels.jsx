import React from 'react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { describePorts } from './portShapes.js';

/** Only the output shape is shown, permanently beside its port. */
export default function PortShapeLabels({ node, shapes }) {
    const getViewGraph = useArchitectureStore((state) => state.getViewGraph);
    const ports = describePorts(getViewGraph(), node, shapes);
    if (ports.output === null) return null;
    return <span className="df-port-shape df-port-shape--output" aria-label={`Output dimensions: ${ports.output}`}>{ports.output}</span>;
}
