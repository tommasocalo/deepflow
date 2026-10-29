/**
 * Python loader for Digits 8×8 (docs/edu/SPEC.md §6.11.1): scikit-learn's load_digits() returns the same 1,797
 * images in the same order as the browser copy, so split_indices gives the same split.
 */
import { joinBlocks, mergeImports, PY_COMMON_IMPORTS, PY_LOADERS, PY_PRNG, PY_SPLIT, PY_TORCH_IMPORTS, pySplit } from './common.js';

export function digitsPythonLoader(spec, { split = {} } = {}) {
    const { val, test } = pySplit(split);
    const build = `def build_dataloaders(batch_size, data_seed, train_seed):
    """Digits 8x8 from scikit-learn: pixel values 0-16 divided by 16, shaped (N, 1, 8, 8) like the
    browser's Input, then split exactly as in the browser."""
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 1, 8, 8)
    splits = split_indices(len(y), ${val}, ${test}, data_seed)
    return make_loaders(X.astype(np.float32), y.astype(np.int64), splits, batch_size, train_seed)
`;
    return {
        imports: mergeImports(PY_COMMON_IMPORTS, PY_TORCH_IMPORTS, ['from sklearn.datasets import load_digits']),
        source: joinBlocks(PY_PRNG, PY_SPLIT, PY_LOADERS, build),
        task: 'classification',
        numClasses: spec.target.size,
        classNames: [...spec.target.classes],
        comment: `${spec.name}: 1,797 grey 8x8 images of handwritten digits, pixels divided by 16. ${spec.attribution}.`,
    };
}
