import React from 'react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { describePorts } from './portShapes.js';

/** Dimensions centered over the wires, controlled by this pane's View setting. */
export default function PortShapeLabels({ node, shapes, hideInput = false }) {
    const getViewGraph = useArchitectureStore((state) => state.getViewGraph);
    const showDimensions = useArchitectureStore((state) => state.showDimensions);
    if (showDimensions === false) return null;
    const ports = describePorts(getViewGraph(), node, shapes);
    return <>
        {!hideInput && ports.input !== null && <span className="df-port-shape df-port-shape--input" aria-label={`Input dimensions: ${ports.input}`}>{ports.input}</span>}
        {ports.output !== null && <span className="df-port-shape df-port-shape--output" aria-label={`Output dimensions: ${ports.output}`}>{ports.output}</span>}
    </>;
}
