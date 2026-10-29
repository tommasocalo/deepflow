import DatasetDetail from './DatasetDetail.jsx';
import DatasetList from './DatasetList.jsx';
import { DATASETS, getDataset } from './registry.js';
import { useEduDataStore } from './store.js';
import './data.css';
// The trainer store registers bridge source 'trainer' on import; the Data tab is one of its mount points.
import '../trainer/store.js';

/**
 * The Data tab (docs/edu/SPEC.md §3.4): the dataset list on the left, the selected dataset on the right. Selecting a
 * row only shows the dataset; "Use this dataset" binds it to the workspace and fits the model's Input and Output.
 */
export default function DataWorkspace() {
    const viewId = useEduDataStore((state) => state.viewId);
    const boundId = useEduDataStore((state) => state.datasetId);
    const setView = useEduDataStore((state) => state.setView);
    const spec = getDataset(viewId) ?? getDataset(boundId) ?? DATASETS[0];
    return (
        <div className="edu-data">
            <DatasetList viewId={spec.id} boundId={boundId} onSelect={setView} />
            <DatasetDetail key={spec.id} spec={spec} />
        </div>
    );
}
