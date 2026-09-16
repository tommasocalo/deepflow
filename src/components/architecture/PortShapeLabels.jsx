import React from 'react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { describePorts } from './portShapes.js';

/** Labels sit beside connection ports without affecting the card's size. */
export default function PortShapeLabels({ node, shapes }) {
    const getViewGraph = useArchitectureStore((state) => state.getViewGraph);
    const ports = describePorts(getViewGraph(), node, shapes);
    return <>
        {ports.input !== null && <span className="df-port-shape df-port-shape--input" aria-label={`Input dimensions: ${ports.input}`}><small>in</small>{ports.input}</span>}
        {ports.output !== null && <span className="df-port-shape df-port-shape--output" aria-label={`Output dimensions: ${ports.output}`}><small>out</small>{ports.output}</span>}
    </>;
}
