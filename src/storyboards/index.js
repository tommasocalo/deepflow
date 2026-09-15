/**
 * Storyboards: pre-built scenarios that exercise the headline primitives.
 * Each storyboard takes the running stores and populates them with a
 * fully wired example. Used for demos, onboarding, and the user study.
 */
import { resnetStoryboard } from './resnet.js';
import { dannStoryboard } from './dann.js';
import { ganStoryboard } from './gan.js';

export const STORYBOARDS = [
    {
        id: 'resnet',
        label: 'ResNet stages',
        summary: 'Parametric Repeat node: one ConvBlock template, three stages with channel doubling.',
        run: resnetStoryboard,
    },
    {
        id: 'dann',
        label: 'DANN',
        summary: 'Domain-adversarial network with a gradient-reversal layer on the domain branch.',
        run: dannStoryboard,
    },
    {
        id: 'gan',
        label: 'GAN',
        summary: 'Two optimizers, two losses, two alternating phases (G step / D step).',
        run: ganStoryboard,
    },
];
