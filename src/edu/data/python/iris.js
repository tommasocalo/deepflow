/**
 * Python loader for Iris (docs/edu/SPEC.md §6.11.1): scikit-learn's load_iris() holds the same 150 rows as the
 * browser, so split_indices gives the same split. Standardisation uses the training rows only.
 */
import { joinBlocks, mergeImports, PY_COMMON_IMPORTS, PY_LOADERS, PY_PRNG, PY_SPLIT, PY_TORCH_IMPORTS, pySplit } from './common.js';

export function irisPythonLoader(spec, { split = {} } = {}) {
    const { val, test } = pySplit(split);
    const build = `def build_dataloaders(batch_size, data_seed, train_seed):
    """Iris from scikit-learn, split exactly as in the browser, then standardised.
    The mean and standard deviation come from the TRAINING split only: computing them on all 150
    flowers would leak information about the validation flowers into training."""
    X, y = load_iris(return_X_y=True)
    train_idx, val_idx, test_idx = split_indices(len(y), ${val}, ${test}, data_seed)
    mean = X[train_idx].mean(axis=0)  # training split only
    std = np.maximum(X[train_idx].std(axis=0), 1e-12)  # population std (ddof 0), training split only
    X = (X - mean) / std
    return make_loaders(X.astype(np.float32), y.astype(np.int64), (train_idx, val_idx, test_idx), batch_size, train_seed)
`;
    return {
        imports: mergeImports(PY_COMMON_IMPORTS, PY_TORCH_IMPORTS, ['from sklearn.datasets import load_iris']),
        source: joinBlocks(PY_PRNG, PY_SPLIT, PY_LOADERS, build),
        task: 'classification',
        numClasses: spec.target.size,
        classNames: [...spec.target.classes],
        comment: `${spec.name}: 150 flowers, 4 measurements in cm, 3 species. ${spec.attribution} `
            + 'Features are standardised with the mean and std of the training split only.',
    };
}
