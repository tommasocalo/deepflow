/**
 * One-shot focus hand-over between views. Leaving a lesson unmounts the control that had focus, so the exit paths
 * ask the Learn home to continue from its heading. An explicit request (instead of "focus whatever mounts next")
 * keeps a first page load, and React StrictMode's repeated effects, from moving focus.
 */
let learnHeadingRequested = false;

export function requestLearnHeadingFocus() {
    learnHeadingRequested = true;
}

/** True once per request. */
export function takeLearnHeadingFocus() {
    const requested = learnHeadingRequested;
    learnHeadingRequested = false;
    return requested;
}
