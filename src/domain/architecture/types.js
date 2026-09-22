/**
 * @typedef {Object} Position
 * @property {number} x
 * @property {number} y
 *
 * @typedef {'int' | 'float' | 'bool' | 'string' | 'tuple' | 'enum'} ParamType
 *
 * @typedef {Object} ParamSpec
 * @property {string} name
 * @property {ParamType} type
 * @property {string} description
 * @property {*} [default]
 * @property {string[]} [options]
 * @property {boolean} [required]
 * @property {{kind: 'input-dimension', inputId: string, axis: number}} [derived] Read-only dimension from the named template input; never an editable instance binding.
 *
 * @typedef {Object} LibraryEntry
 * @property {string} key
 * @property {string} name
 * @property {string} category
 * @property {string} description
 * @property {string | null} pyTorchFn
 * @property {ParamSpec[]} parameters
 * @property {string[]} [aliases]
 *
 * @typedef {Object} LayerNode
 * @property {string} id
 * @property {'layer'} kind
 * @property {string} libraryKey
 * @property {string} label
 * @property {Position} position
 * @property {Object<string, *>} parameters
 *
 * @typedef {Object} SuperNodeInstance
 * @property {string} id
 * @property {'supernode'} kind
 * @property {string} templateId           Reference to the template definition
 * @property {string} label                 Per-instance display label
 * @property {Position} position
 * @property {Object<string, *>} [parameterValues]   Bound values for the template's free parameters (literal or expression strings)
 * @property {{count: number|string, mode: string, iterationVariable: string, bindings: Object<string, *>, singleValues: Object<string, *>}} [repeatSettings] Remembered repetition plus its iteration-0 projection, used to restore formulas unless the Single values were edited.
 *
 * @typedef {Object} RepeatNode
 * @property {string} id
 * @property {'repeat'} kind
 * @property {string} templateId           Template to repeat
 * @property {string} label
 * @property {Position} position
 * @property {number | string} count       Iterations, an expression in the parent parameters, or 'auto' to use incoming collection length for parallel repeat.
 * @property {'sequential' | 'parallel'} [mode] Missing values retain legacy sequential behavior. Parallel branches produce a collection of tensors.
 * @property {string} iterationVariable    Name bound to the iteration index (default: 'i')
 * @property {Object<string, *>} bindings  Per-iteration formulas keyed by free-param name. Each value can be a scalar (literal or expression in the iteration variable) or an array [v0, v1, ...] of explicit per-iteration values.
 *
 * @typedef {LayerNode | SuperNodeInstance | RepeatNode} ArchNode
 *
 * @typedef {'none' | 'detach' | 'stop' | 'reverse'} GradientOp
 *
 * `none`     — gradients flow normally
 * `detach`   — backward returns zero through this edge (LoRA bypass, frozen)
 * `stop`     — same as detach in semantics; presented separately so
 *              the user can express "stop_gradient" intent
 * `reverse`  — gradient sign flipped (× -lambda) — DANN's GRL
 *
 * @typedef {Object} ArchEdge
 * @property {string} id
 * @property {string} source
 * @property {string} target
 * @property {string} [sourceHandle]
 * @property {string} [targetHandle]
 * @property {GradientOp} [gradientOp]
 * @property {number} [gradientOpLambda]                  for `reverse`: scaling factor (default 1.0)
 *
 * @typedef {Object} ArchitectureGraph
 * @property {Object<string, ArchNode>} nodes
 * @property {ArchEdge[]} edges
 *
 * --------------------------------------------------------------------------
 *
 * Templates are the parametric definition behind a SuperNodeInstance. Their
 * internal subgraph contains layer nodes plus distinguished boundary
 * blocks — `deepflow.template.Input` and `deepflow.template.Output` — that
 * mark how the supernode's external edges hook up to its internals. Multiple
 * input boundaries use their node IDs as target handles on the parent node.
 *
 * @typedef {Object} ArchitectureTemplate
 * @property {string} id
 * @property {string} name
 * @property {string} description
 * @property {ArchitectureGraph} graph        Internal subgraph (incl. boundary nodes)
 * @property {ParamSpec[]} [freeParameters]   Parameters exposed to instances and repeats
 * @property {string} [standardKey]           Editable catalog family
 * @property {number} [standardVersion]       Catalog generation; old definitions remain attached to saved graphs
 * @property {string} [componentStyleKey]     Visual family retained by independent copies
 * @property {string} [sourceTemplateId]      Definition from which a copy was derived
 */

export {};
