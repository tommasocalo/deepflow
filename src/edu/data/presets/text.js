/**
 * Export-only text preset (docs/edu/SPEC.md §5.1, §5.2, §6.11.1): Tiny Shakespeare, for character-level language
 * models (module 7).
 *
 * Nothing is downloaded in the browser and no text is shown: train.py downloads Karpathy's input.txt when it runs,
 * so `load` is deliberately absent. The numbers below were measured on that file: 1,115,394 characters, 65 distinct
 * characters, and int(0.9 · n) = 1,003,854 characters for training and 111,540 for validation (as in nanoGPT).
 */
import { resolveOptions } from '../options.js';
import { pyNumber, pyString } from '../python/common.js';

export const TINY_SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
const CHARACTERS = 1115394;
const TRAIN_CHARACTERS = Math.floor(0.9 * CHARACTERS);
const VOCABULARY = 65;
const grouped = (n) => n.toLocaleString('en-US');

const IMPORTS = ['import os', 'import urllib.request', 'import torch'];

/**
 * The PythonLoader of §6.11.1: random windows of `blockSize` token ids. Each "epoch" is `stepsPerEpoch` training
 * batches drawn with a seeded generator; validation uses 50 fixed batches, so its loss is comparable across epochs.
 */
export function textPythonLoader(spec, { options = {} } = {}) {
    const { blockSize } = resolveOptions(spec, options);
    const steps = spec.exportDefaults.stepsPerEpoch;
    const source = `DATA_URL = ${pyString(TINY_SHAKESPEARE_URL)}
BLOCK_SIZE = ${pyNumber(blockSize)}  # characters per example: the length T of your model's Input
TRAIN_BATCHES = ${pyNumber(steps)}  # random windows per "epoch": visiting every window would take far longer
VAL_BATCHES = 50


def load_text(path="data/tinyshakespeare.txt"):
    """Download Tiny Shakespeare (about 1.1 MB) the first time and keep a copy in data/."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as file:
        return file.read()


class Vocabulary:
    """The distinct characters of the text, sorted. A character's position in the list is its token id."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # character -> token id
        self.itos = dict(enumerate(self.chars))  # token id -> character

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class RandomWindows:
    """Batches of random windows from one long sequence of token ids: x = data[i : i + T] and
    y = data[i + 1 : i + T + 1], so y[t] is the character that comes after x[t]. One pass yields \`batches\`
    batches of shape (batch_size, T). With fixed=True every pass yields the same windows, so the validation
    loss of one epoch can be compared with the next."""

    def __init__(self, data, vocab, block_size, batch_size, batches, seed, fixed=False):
        self.data, self.vocab = data, vocab
        self.block_size, self.batch_size, self.batches = block_size, batch_size, batches
        self.seed, self.fixed = seed, fixed
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.batches

    def __iter__(self):
        if self.fixed:
            self.generator.manual_seed(self.seed)
        T = self.block_size
        for _ in range(self.batches):
            starts = torch.randint(len(self.data) - T, (self.batch_size,), generator=self.generator).tolist()
            x = torch.stack([self.data[i:i + T] for i in starts])
            y = torch.stack([self.data[i + 1:i + T + 1] for i in starts])
            yield x, y


def build_dataloaders(batch_size, data_seed, train_seed):
    """Tiny Shakespeare as token ids: the first 90% of the text for training (${grouped(TRAIN_CHARACTERS)} characters),
    the last 10% for validation (${grouped(CHARACTERS - TRAIN_CHARACTERS)}), as in nanoGPT. There is no test split."""
    text = load_text()
    vocab = Vocabulary(text)  # ${VOCABULARY} characters: 52 letters, newline, space, punctuation, "$", "&" and "3"
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_loader = RandomWindows(data[:n], vocab, BLOCK_SIZE, batch_size, TRAIN_BATCHES, seed=train_seed)
    val_loader = RandomWindows(data[n:], vocab, BLOCK_SIZE, batch_size, VAL_BATCHES, seed=data_seed, fixed=True)
    return train_loader, val_loader, None
`;
    return {
        imports: [...IMPORTS],
        source,
        task: 'next-token',
        vocabNote: `${VOCABULARY} characters`,
        comment: `${spec.name}: ${grouped(CHARACTERS)} characters of Shakespeare's plays, ${VOCABULARY} distinct characters. `
            + `Each example is a window of ${blockSize} character ids; the target is the same window shifted by one. ${spec.attribution}.`,
    };
}

const tinyShakespeare = {
    id: 'tiny-shakespeare',
    name: 'Tiny Shakespeare',
    description: 'About 1.1 million characters from Shakespeare\'s plays. A model learns to predict the next character, then writes new text in the same style.',
    kind: 'text',
    training: 'export',
    input: { shape: '(64,)', dtype: 'int64' },
    target: { kind: 'next-token', size: VOCABULARY },
    size: {
        total: CHARACTERS,
        train: TRAIN_CHARACTERS,
        val: CHARACTERS - TRAIN_CHARACTERS,
        note: `${grouped(CHARACTERS)} characters: the first ${grouped(TRAIN_CHARACTERS)} for training, the last ${grouped(CHARACTERS - TRAIN_CHARACTERS)} for validation`,
    },
    licence: {
        name: 'Public-domain text (Shakespeare); file from karpathy/char-rnn (MIT)',
        url: 'https://github.com/karpathy/char-rnn',
        verified: true,
        note: 'MIT as stated in the char-rnn README, which has no separate licence file. Not bundled: train.py downloads input.txt (about 1.1 MB) when it runs',
    },
    attribution: 'Tiny Shakespeare, from Karpathy\'s char-rnn',
    citation: [
        'A. Karpathy (2015). char-rnn: multi-layer recurrent neural networks for character-level language models. https://github.com/karpathy/char-rnn',
        'A. Karpathy (2015). The Unreasonable Effectiveness of Recurrent Neural Networks. https://karpathy.github.io/2015/05/21/rnn-effectiveness/',
    ],
    options: { blockSize: { default: 64, min: 8, max: 1024, step: 8, label: 'Block size (characters)' } },
    exportDefaults: { epochs: 10, batchSize: 32, stepsPerEpoch: 200 },
    /** One sentence shown in the Data tab instead of a data preview (§3.4). */
    schematic: `One long text of ${grouped(CHARACTERS)} characters written with ${VOCABULARY} distinct characters. Each example is a window of 64 character ids, and the target is the same window shifted one character to the left.`,
    preview: 'none',
    featureNames: null,
    python: (ctx = {}) => textPythonLoader(tinyShakespeare, ctx),
};

export default [tinyShakespeare];
