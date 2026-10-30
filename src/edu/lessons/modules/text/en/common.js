/**
 * The misconception catalogue (docs/edu/SPEC.md §4.9). Ids are stable; every prediction diagnosis, palette
 * distractor, quiz or warm-up distractor and step diagnostic names one of them. The lesson panel shows the text
 * as "A common mix-up: …", and the quiz appends it to the chosen option's explanation, so each entry states the
 * general rule rather than repeating the option.
 */
export default {
    // Counting parameters and sizes
    'mis.forgot-bias': 'Each output unit also has one bias, so add the number of outputs.',
    'mis.added-not-multiplied': 'Sizes multiply: a 28 × 28 image has 784 pixels, and a Linear block has inputs × outputs weights.',
    'mis.one-axis-only': 'Flatten multiplies every axis after the batch axis, for MNIST 1 × 28 × 28.',
    'mis.one-map-only': 'Flatten multiplies all three axes: channels × height × width.',
    'mis.channels-times-height': 'Flatten multiplies all three axes: channels × height × width.',
    'mis.weights-per-position': 'A filter\'s weights are shared across positions, so they do not grow with the image size.',
    'mis.one-filter-only': 'Each of the 8 filters has its own 3 × 3 weights and its own bias.',
    'mis.forgot-batchnorm': 'BatchNorm2d(32) learns a scale and a shift per channel: 64 parameters.',
    'mis.conv-bias-when-disabled': 'These convolutions have bias off; BatchNorm\'s shift does that job.',
    'mis.one-number-per-token': 'Each token gets a whole learned vector, one number per feature, not a single number.',

    // Inputs, outputs and labels
    'mis.target-in-input': 'The label (here the species) is what the model predicts; it is not an input feature.',
    'mis.classes-as-features': '3 is the number of species, the outputs. The inputs are the 4 measurements.',
    'mis.examples-as-features': '150 is the number of flowers, the examples. The model sees one flower at a time, with 4 features.',
    'mis.one-output-multiclass': 'With K classes the model needs K outputs, one score per class.',
    'mis.bce-for-multiclass': 'BCE scores one yes/no output. With K classes, use cross-entropy on K scores.',
    'mis.mse-for-classes': 'MSE measures distances between numbers. Class labels are categories, not amounts.',
    'mis.softmax-before-ce': 'Cross-entropy already applies softmax. A Softmax block before it squashes the scores twice and slows learning: output raw scores (logits).',
    'mis.sigmoid-before-bce': 'DeepFlow\'s BCE is BCEWithLogitsLoss, which applies the sigmoid itself. Output the raw score.',
    'mis.confusion-axes': 'Rows hold the true class and columns the predicted class. A correct answer has the same class in both, so it sits on the diagonal.',

    // Validation and scaling
    'mis.validation-trains': 'Validation examples never update the weights; they only measure the model.',
    'mis.validation-for-speed': 'The split exists for an honest measurement, not for speed.',
    'mis.scaling-leak': 'Statistics fitted on validation data leak information about it into training, so the validation score looks better than it is.',

    // Linear maps and non-linearities
    'mis.neuron-bends': 'One neuron computes a weighted sum, so its boundary w₁x₁ + w₂x₂ + b = 0 is always a straight line.',
    'mis.stacked-linear-bends': 'Linear blocks in a row still compute one linear function. Only a non-linearity between them lets the boundary bend.',
    'mis.second-linear-ignored': 'Both blocks train, but together they still form one linear map.',
    'mis.stacked-share-weights': 'Each Linear block has its own weights. The collapse comes from composing two linear maps.',
    'mis.piecewise-linear-is-linear': 'A function made of straight pieces is non-linear when it bends: ReLU bends at 0.',
    'mis.nonlinear-squash': 'Tanh and sigmoid curve their input: doubling z does not double the output, so they are non-linear.',
    'mis.activation-at-output': 'A ReLU right before the output clips negative scores to 0. Put activations between Linear blocks.',
    'mis.too-few-edges': 'A closed region needs at least three straight edges, like a triangle. One or two edges leave it open.',
    'mis.more-units-needed': 'More units also work, but they are not the minimum: a triangle needs only three edges.',
    'mis.unit-draws-circle': 'A ReLU unit is linear on each side of one line, so it adds a straight edge, not a curve.',
    'mis.unit-memorises-point': 'Hidden units do not store training points. Each one adds a line along which the output can bend.',
    'mis.matmul-keeps-width': 'MatMul replaces the last axis: (B, 2) × (2, 3) gives (B, 3).',

    // Shapes
    'mis.mismatched-widths': 'The second Linear\'s Input must equal the first Linear\'s Output.',
    'mis.tool-fixes-shapes': 'DeepFlow reports a size mismatch but never changes your model for you. You fix the field.',
    'mis.missing-channel-axis': 'Grey images still have a channel axis of size 1: (B, 1, 28, 28).',
    'mis.missing-batch-axis': 'Models process batches: the first axis is the batch size B.',
    'mis.batch-axis-meaning': 'B is the batch axis: how many examples pass through the model together.',
    'mis.flattened-too-early': 'Images arrive as (B, 1, 28, 28). Flatten is a block you add; it is not part of the data.',
    'mis.channels-last': 'PyTorch puts channels before height and width: (B, C, H, W).',
    'mis.normalize-changes-pixels': 'Normalize only rescales pixel values: it subtracts a mean and divides by a standard deviation. The image keeps its size.',
    'mis.loss-starts-fixed': 'The first loss depends on the number of classes: about ln C, so 1.10 for 3 classes.',

    // Convolutions and pooling
    'mis.channels-unchanged': 'A convolution outputs one channel per filter.',
    'mis.forgot-padding': 'That would be right without padding. Padding 1 keeps the size at 28.',
    'mis.kernel-is-output': 'The kernel is the filter\'s size. The output map keeps the image\'s height and width, minus what the border loses.',
    'mis.padding-no-effect': 'Without padding, the filter cannot be centred on the border pixels, so the map shrinks.',
    'mis.minus-kernel': 'With stride 1 and no padding the size is n − k + 1, not n − k.',
    'mis.in-channels-mismatch': 'A convolution\'s In channels must equal the previous convolution\'s Out channels.',
    'mis.missing-flatten': 'Linear acts on the last axis only. Add Flatten after the last pooling block.',
    'mis.pooling-halves-channels': 'Pooling works on each channel separately: it shrinks height and width and keeps every channel.',
    'mis.pool-stride-one': 'MaxPool2d\'s stride defaults to its kernel size, so a 2 × 2 pool halves height and width.',
    'mis.conv-no-weights': 'Each filter has learned weights and a bias, like a Linear block.',
    'mis.conv-sees-less': 'A CNN still looks at every pixel. It reuses the same small filters at every position.',

    // Losses and the training recipe
    'mis.zero-initial-loss': 'An untrained model guesses, so its loss starts high, near −ln(1/C).',
    'mis.loss-is-error-rate': 'Loss is not the error rate. For uniform guesses it is −ln(1/C).',
    'mis.loss-is-probability': 'Loss is −ln of the probability of the right class, not the probability itself.',
    'mis.loss-is-count': 'Loss is −ln(probability of the right answer), here −ln(1/C).',
    'mis.low-initial-loss-good': 'An untrained model cannot know the answers. A first loss far below ln C means it sees something it should not, such as the labels.',
    'mis.initial-loss-arbitrary': 'The first loss is predictable: about ln C for C balanced classes.',
    'mis.bigger-lr-faster': 'A larger step is not always faster. Past a point the updates overshoot and the loss jumps around.',
    'mis.smaller-lr-safer': 'A tiny learning rate is stable but slow: in 60 epochs the loss barely moves.',
    'mis.bigger-model-fixes-training': 'A bigger model does not fix unstable updates; a smaller learning rate does.',
    'mis.training-loss-rises': 'Training loss keeps falling, because the optimizer lowers it directly. The held-out loss is the one that can rise.',
    'mis.more-epochs-always-better': 'More epochs do not always help. After the best epoch the model starts to memorise noise.',
    'mis.underfitting-confusion': 'Underfitting means the model cannot even fit its training data. Here training accuracy is near 100%.',
    'mis.training-loss-is-enough': 'A low training loss alone proves little. Judge a model on data it never trained on.',
    'mis.dropout-p-meaning': 'In PyTorch, p is the share of hidden values set to 0, not the share kept: p = 0.9 drops nine in ten.',
    'mis.dropout-at-eval': 'Dropout is random and only active in training mode. model.eval() turns it off.',
    'mis.eval-freezes-weights': 'model.eval() does not stop learning: torch.no_grad() and skipping optimizer.step() do that.',
    'mis.eval-shrinks-model': 'model.eval() keeps every weight. It only changes how Dropout and BatchNorm behave.',

    // Components, repeats and residual connections
    'mis.group-changes-model': 'Group does not change what the model computes or how it trains. It wraps the blocks in a reusable component.',
    'mis.shared-structure-shared-weights': 'Repeat copies the structure, not the numbers: every copy has its own weights, initialised and trained separately.',
    'mis.repeat-splits-params': 'Copies do not split one set of weights between them. Each copy has a full set of its own.',
    'mis.add-shapes': 'Both inputs of Add must have the same shape.',
    'mis.add-concatenates': 'Add sums element by element. Joining tensors side by side is what Concat does.',
    'mis.skip-doubles-gradient': 'The skip adds an identity term to the block\'s gradient, ∂y/∂x = I + ∂F/∂x. It does not double it.',
    'mis.skip-blocks-gradient': 'Gradients flow back through every differentiable path, the skip included.',
    'mis.gradient-view-magnitudes': 'The Gradients view is static: it shows which paths can carry gradients, not their size during training.',

    // Attention and language models
    'mis.one-prediction-per-sequence': 'A language model predicts the next token at every position, not once per sequence.',
    'mis.ids-not-scores': 'The output holds a score for every token in the vocabulary at each position.',
    'mis.axes-swapped': 'Positions come before scores: (B, T, V), here (B, 64, 65).',
    'mis.attention-knows-order': 'Attention compares tokens by content. Without position vectors, shuffling the input only shuffles the output.',
    'mis.attention-sees-future': 'A language model must not see the character it predicts: the causal mask hides every later position.',
    'mis.attention-only-self': 'Attention mixes information from many positions. The mask only hides the later ones.',
    'mis.mask-hides-self': 'The mask keeps the current position: each token attends to itself and to earlier tokens.',
    'mis.mask-is-position': 'The mask and the position vectors do different jobs: the mask hides the future, positions give order.',
    'mis.heads-multiply-width': 'The heads share d_model: each one gets d_model / n_heads features.',
    'mis.heads-must-divide': 'n_heads must divide d_model evenly, so every head gets the same whole number of features.',
    'mis.head-width-vs-count': '64 is the width of each head. GPT-2 small splits d_model 768 into 768 / 64 = 12 heads.',
    'mis.temperature-creativity': 'Temperature only rescales the scores before softmax. Lower values make the likeliest characters even likelier.',
    'mis.temperature-changes-weights': 'Sampling settings never change the trained weights.',
};
