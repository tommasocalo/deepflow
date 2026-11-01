/**
 * Export-only image presets (docs/edu/SPEC.md §5.1, §5.2, §6.11.1): MNIST, Fashion-MNIST and CIFAR-10.
 *
 * They carry the facts a learner needs to build a model (input shape, classes, sizes), the licence and the
 * normalisation constants. Nothing is downloaded in the browser and no sample images are shown: `train.py` downloads
 * the official files with torchvision when it runs, so `load` is deliberately absent.
 */
import { resolveOptions } from '../options.js';
import { pyNumber } from '../python/common.js';

const DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
const FASHION = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
const CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

/** Python tuple with 4 decimals, as the constants are usually quoted: (0.1307,) or (0.4914, 0.4822, 0.4465). */
export const pyTuple4 = (values) => `(${values.map((value) => value.toFixed(4)).join(', ')}${values.length === 1 ? ',' : ''})`;
const grouped = (n) => n.toLocaleString('en-US');

const IMPORTS = [
    'import torch',
    'from torch.utils.data import DataLoader, random_split',
    'from torchvision import datasets, transforms',
];

/**
 * The PythonLoader of §6.11.1 for a torchvision dataset: the official training set minus `valSize` validation
 * images (a seeded random_split), and the official test set, which train.py evaluates once at the end.
 */
export function torchvisionPythonLoader(spec, { options = {} } = {}) {
    const { valSize } = resolveOptions(spec, options);
    const [channels, height, width] = spec.inputDims;
    const { mean, std } = spec.normalization;
    const what = channels === 1
        ? `the mean pixel value of the ${spec.name} training images and divide by their standard deviation`
        : `the mean of each colour channel over the ${spec.name} training images and divide by its standard deviation`;
    const source = `VAL_SIZE = ${pyNumber(valSize)}  # training images held out for validation


def build_dataloaders(batch_size, data_seed, train_seed):
    """${spec.name} from torchvision, downloaded to data/ the first time (${spec.downloadNote}).

    The official ${grouped(spec.size.train)} training images are split into ${grouped(spec.size.train - valSize)} for training and ${grouped(valSize)} for
    validation (seeded with data_seed). The official ${grouped(spec.size.test)} test images are used once, at the end.
    The Normalize constants are the usual ones, computed over all ${grouped(spec.size.train)} official training images,
    the ${grouped(valSize)} validation images included: a small, common shortcut. Iris and the toy datasets fit
    their scaling on the training split only."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # image -> float tensor (${channels}, ${height}, ${width}) with values in [0, 1]
        # Subtract ${what}:
        transforms.Normalize(${pyTuple4(mean)}, ${pyTuple4(std)}),
    ])
    root = "data"
    train_full = datasets.${spec.torchvisionName}(root, train=True, download=True, transform=transform)
    test_set = datasets.${spec.torchvisionName}(root, train=False, download=True, transform=transform)
    train_set, val_set = random_split(train_full, [len(train_full) - VAL_SIZE, VAL_SIZE],
                                      generator=torch.Generator().manual_seed(data_seed))
    workers = 2  # background processes that decode images while the model trains
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                              generator=torch.Generator().manual_seed(train_seed))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader
`;
    return {
        imports: [...IMPORTS],
        source,
        task: 'classification',
        numClasses: spec.target.size,
        classNames: [...spec.target.classes],
        comment: `${spec.name}: ${spec.schematic} ${spec.attribution}.`,
    };
}

/**
 * `normalization` holds the usual torchvision constants. They are computed over all official training images,
 * including the `valSize` images train.py holds out for validation: a small, common shortcut, unlike the training-
 * split-only scaling of Iris and the toy sets (M1). M3 s7's reveal and the loader's docstring say so.
 */
function imageSpec({ id, name, torchvisionName, description, schematic, dims, classes, train, test, normalization, licence, attribution, citation, exportDefaults, downloadNote }) {
    const spec = {
        id,
        name,
        description,
        kind: 'image',
        training: 'export',
        input: { shape: `(${dims.join(', ')})`, dtype: 'float32' },
        target: { kind: 'classes', classes, size: classes.length },
        size: { train, test, note: `${grouped(train)} training + ${grouped(test)} test images` },
        licence,
        attribution,
        citation,
        options: { valSize: { default: 5000, min: 1000, max: 10000, step: 500, label: 'Validation images' } },
        /** Training images train.py holds out for validation (the Data tab's split text reads it). */
        valSize: 5000,
        normalization,
        exportDefaults,
        /** One sentence shown in the Data tab instead of a data preview (§3.4). */
        schematic,
        preview: 'none',
        featureNames: null,
        torchvisionName,
        inputDims: dims,
        downloadNote,
        python: (ctx = {}) => torchvisionPythonLoader(spec, ctx),
    };
    return spec;
}

export default [
    imageSpec({
        id: 'mnist',
        name: 'MNIST',
        torchvisionName: 'MNIST',
        description: '70,000 grey 28 × 28 images of handwritten digits in 10 classes: the classic first image dataset.',
        schematic: '60,000 training and 10,000 test images of handwritten digits, 28 × 28 pixels, grey (1 channel).',
        dims: [1, 28, 28],
        classes: DIGITS,
        train: 60000,
        test: 10000,
        normalization: { mean: [0.1307], std: [0.3081] },
        licence: {
            name: 'Not bundled; no licence is stated by the authors and other sources disagree',
            url: 'http://yann.lecun.com/exdb/mnist/',
            verified: false,
            note: 'torchvision downloads the original files when train.py runs',
        },
        attribution: 'LeCun, Cortes & Burges, The MNIST database',
        citation: [
            'Y. LeCun, C. Cortes, C. J. C. Burges. The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/',
            'Y. LeCun, L. Bottou, Y. Bengio, P. Haffner (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.',
        ],
        exportDefaults: { epochs: 5, batchSize: 64 },
        downloadNote: 'about 12 MB',
    }),
    imageSpec({
        id: 'fashion-mnist',
        name: 'Fashion-MNIST',
        torchvisionName: 'FashionMNIST',
        description: 'Clothes instead of digits, in the same format as MNIST: 70,000 grey 28 × 28 images in 10 classes.',
        schematic: '60,000 training and 10,000 test images of clothing, 28 × 28 pixels, grey (1 channel).',
        dims: [1, 28, 28],
        classes: FASHION,
        train: 60000,
        test: 10000,
        normalization: { mean: [0.2860], std: [0.3530] },
        licence: {
            name: 'MIT, Copyright © 2017 Zalando SE',
            url: 'https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE',
            verified: true,
            note: 'Not bundled: torchvision downloads it when train.py runs',
        },
        attribution: 'Xiao, Rasul & Vollgraf (2017), Fashion-MNIST (MIT)',
        citation: ['H. Xiao, K. Rasul, R. Vollgraf (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747.'],
        exportDefaults: { epochs: 5, batchSize: 64 },
        downloadNote: 'about 30 MB',
    }),
    imageSpec({
        id: 'cifar10',
        name: 'CIFAR-10',
        torchvisionName: 'CIFAR10',
        description: '60,000 small colour photos, 32 × 32 pixels, of 10 kinds of objects and animals, such as airplanes, cats and trucks.',
        schematic: '50,000 training and 10,000 test colour photos, 32 × 32 pixels, 3 channels (red, green, blue).',
        dims: [3, 32, 32],
        classes: CIFAR10,
        train: 50000,
        test: 10000,
        // The per-channel std over all training pixels. The also-common (0.2023, 0.1994, 0.2010) is a mean of
        // per-image stds and is not used (docs/edu/SPEC.md §5.1).
        normalization: { mean: [0.4914, 0.4822, 0.4465], std: [0.2470, 0.2435, 0.2616] },
        licence: {
            name: 'Not bundled; no licence is stated by the authors',
            url: 'https://www.cs.toronto.edu/~kriz/cifar.html',
            verified: false,
            note: 'torchvision downloads it (about 170 MB) when train.py runs',
        },
        attribution: 'Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images',
        citation: ['A. Krizhevsky (2009). Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto.'],
        exportDefaults: { epochs: 10, batchSize: 128 },
        downloadNote: 'about 170 MB',
    }),
];
