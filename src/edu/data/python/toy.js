/**
 * Python loaders for the 2D toy datasets (docs/edu/SPEC.md §6.11.1). Each generator is a line-by-line port of
 * data/generators.js; keep the expressions in the same order so the floats match the browser bit for bit.
 */
import {
    joinBlocks, mergeImports, PY_COMMON_IMPORTS, PY_LOADERS, PY_PRNG, PY_SPLIT, PY_TORCH_IMPORTS, pyNumber, pySplit,
} from './common.js';

const POINTS = `def points_to_arrays(points, regression=False):
    """(x1, x2, label) tuples to X (N, 2) float64 and y (int64 labels, or float64 targets)."""
    X = np.array([(p[0], p[1]) for p in points], dtype=np.float64)
    y = np.array([p[2] for p in points], dtype=np.float64 if regression else np.int64)
    return X, y
`;

/** Torch-free generator definitions, keyed by dataset id. */
export const PY_GENERATORS = {
    blobs: `def generate_blobs(n, noise, seed):
    """Two Gaussian clouds: label 1 around (2, 2), label 0 around (-2, -2) (TensorFlow Playground, Apache-2.0)."""
    next_ = mulberry32(derive_seed(seed, "data"))
    half = n // 2
    variance = 0.5 + 3.5 * (noise / 0.5)
    std = math.sqrt(variance)
    points = []
    for center, label in ((2, 1), (-2, 0)):
        for _ in range(half):
            x1 = normal(next_, center, std)
            x2 = normal(next_, center, std)
            points.append((x1, x2, label))
    return points_to_arrays(points)
`,
    circle: `def generate_circle(n, noise, seed):
    """A disk (label 1, radius < 2.5) inside a ring (label 0, radius 3.5 to 5) (TensorFlow Playground, Apache-2.0)."""
    next_ = mulberry32(derive_seed(seed, "data"))
    half = n // 2
    points = []
    for low, high in ((0, 2.5), (3.5, 5)):
        for _ in range(half):
            r = uniform(next_, low, high)
            angle = uniform(next_, 0, 2 * math.pi)
            x1 = r * math.sin(angle)
            x2 = r * math.cos(angle)
            noise_x = uniform(next_, -5, 5) * noise
            noise_y = uniform(next_, -5, 5) * noise
            dx = x1 + noise_x
            dy = x2 + noise_y
            points.append((x1, x2, 1 if math.sqrt(dx * dx + dy * dy) < 2.5 else 0))
    return points_to_arrays(points)
`,
    xor: `def generate_xor(n, noise, seed):
    """Label 1 where x1 and x2 have the same sign (TensorFlow Playground, Apache-2.0)."""
    next_ = mulberry32(derive_seed(seed, "data"))
    points = []
    for _ in range(n):
        x1 = uniform(next_, -5, 5)
        x1 += 0.3 if x1 > 0 else -0.3
        x2 = uniform(next_, -5, 5)
        x2 += 0.3 if x2 > 0 else -0.3
        noise_x = uniform(next_, -5, 5) * noise
        noise_y = uniform(next_, -5, 5) * noise
        points.append((x1, x2, 1 if (x1 + noise_x) * (x2 + noise_y) >= 0 else 0))
    return points_to_arrays(points)
`,
    spiral2: `def generate_spiral2(n, noise, seed):
    """Two interleaved spiral arms: label 1, then label 0 rotated by pi (TensorFlow Playground, Apache-2.0)."""
    next_ = mulberry32(derive_seed(seed, "data"))
    half = n // 2
    points = []
    for delta, label in ((0, 1), (math.pi, 0)):
        for i in range(half):
            r = i / half * 5
            t = 1.75 * i / half * 2 * math.pi + delta
            x1 = r * math.sin(t) + uniform(next_, -1, 1) * noise
            x2 = r * math.cos(t) + uniform(next_, -1, 1) * noise
            points.append((x1, x2, label))
    return points_to_arrays(points)
`,
    spiral3: `def generate_spiral3(per_class, noise, seed):
    """Three spiral arms, as in the CS231n neural-network case study (MIT)."""
    next_ = mulberry32(derive_seed(seed, "data"))
    points = []
    for j in range(3):
        for i in range(per_class):
            r = i / (per_class - 1)
            t = 4 * j + 4 * i / (per_class - 1) + normal(next_, 0, 1) * noise
            points.append((r * math.sin(t), r * math.cos(t), j))
    return points_to_arrays(points)
`,
    plane: `def generate_plane(n, noise, seed):
    """Regression target (x1 + x2) / 10 with input noise (TensorFlow Playground, Apache-2.0)."""
    next_ = mulberry32(derive_seed(seed, "data"))
    points = []
    for _ in range(n):
        x1 = uniform(next_, -6, 6)
        x2 = uniform(next_, -6, 6)
        noise_x = uniform(next_, -6, 6) * noise
        noise_y = uniform(next_, -6, 6) * noise
        points.append((x1, x2, (x1 + noise_x + x2 + noise_y) / 10))
    return points_to_arrays(points, regression=True)
`,
};

/** The generator call with the dataset's options, e.g. `generate_blobs(200, 0.1, data_seed)`. */
export function pyGeneratorCall(id, options, seedExpression = 'data_seed') {
    const size = id === 'spiral3' ? options.perClass : options.n;
    return `generate_${id}(${pyNumber(size)}, ${pyNumber(options.noise ?? 0)}, ${seedExpression})`;
}

/** The PythonLoader of §6.11.1 for a toy dataset spec. */
export function toyPythonLoader(spec, { options = {}, split = {} } = {}) {
    const regression = spec.target.kind === 'regression';
    const { val, test } = pySplit(split);
    const sizeText = spec.id === 'spiral3' ? `${options.perClass} points per class` : `${options.n} points`;
    const comment = `${spec.name}: ${sizeText}, noise ${options.noise ?? 0}. ${spec.attribution}. `
        + 'The same points and train/validation split as in the browser.';
    const y = regression ? 'y.astype(np.float32).reshape(-1, 1)' : 'y';
    const build = `def build_dataloaders(batch_size, data_seed, train_seed):
    """${spec.name}: regenerate the browser's points from data_seed, then make the same split."""
    X, y = ${pyGeneratorCall(spec.id, options)}
    splits = split_indices(len(y), ${val}, ${test}, data_seed)
    return make_loaders(X.astype(np.float32), ${y}, splits, batch_size, train_seed)
`;
    return {
        imports: mergeImports(PY_COMMON_IMPORTS, PY_TORCH_IMPORTS),
        source: joinBlocks(PY_PRNG, PY_SPLIT, PY_LOADERS, POINTS, PY_GENERATORS[spec.id], build),
        task: regression ? 'regression' : spec.target.size === 2 ? 'binary' : 'classification',
        ...(regression ? {} : { numClasses: spec.target.size, classNames: [...spec.target.classes] }),
        comment,
    };
}
