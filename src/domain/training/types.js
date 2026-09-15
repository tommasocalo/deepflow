/**
 * @typedef {Object} OptimizerNode
 * @property {string} id
 * @property {'optimizer'} kind
 * @property {import('../architecture/types.js').Position} position
 * @property {string} label
 * @property {'Adam' | 'AdamW' | 'SGD' | 'RMSprop'} type
 * @property {Object<string, *>} hyperparameters     learning_rate, weight_decay, momentum, etc.
 * @property {string[]} parameterGroup               architecture node IDs to train; empty = all parameters
 *
 * @typedef {Object} LossNode
 * @property {string} id
 * @property {'loss'} kind
 * @property {import('../architecture/types.js').Position} position
 * @property {string} label
 * @property {'cross_entropy' | 'mse' | 'bce' | 'l1' | 'custom'} type
 * @property {Object<string, *>} hyperparameters
 * @property {string | null} architectureOutputId     architecture Output node providing predictions
 * @property {string} labelKey                         dataset column with the targets
 * @property {number} weight                           multiplier for combined-loss aggregation
 *
 * @typedef {Object} PhaseNode
 * @property {string} id
 * @property {'phase'} kind
 * @property {import('../architecture/types.js').Position} position
 * @property {string} label
 * @property {string[]} optimizerIds                   active optimizers in this phase
 * @property {string[]} lossIds                        active losses in this phase
 * @property {number} everyNSteps                      run every N global steps; 1 = every step
 * @property {number} stepsPerCycle                    how many steps to run when activated
 *
 * @typedef {OptimizerNode | LossNode | PhaseNode} TrainingNode
 *
 * @typedef {Object} TrainingEdge
 * @property {string} id
 * @property {string} source
 * @property {string} target
 * @property {string} [sourceHandle]
 * @property {string} [targetHandle]
 *
 * @typedef {Object} TrainingGraph
 * @property {Object<string, TrainingNode>} nodes
 * @property {TrainingEdge[]} edges
 */

export {};
