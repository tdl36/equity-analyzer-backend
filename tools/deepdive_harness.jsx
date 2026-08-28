// Deterministic golden-render harness: the three artifacts from the reviewed
// Deere fixture, with no backend, auth or app chrome. This is what makes a
// side-by-side against best_proven_outputs/DE_v29_*.pdf possible.
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { DeepDiveArtifact, preflightPages } from '../src/deepdive';
import fixture from '../fixtures/deepdive_de_golden.json';

const q = new URLSearchParams(location.search);
const view = q.get('view') || 'onepager';
const template = q.get('template') || 'notebook';
const root = document.getElementById('root');
const run = { master: fixture.master, onepager: fixture.onepager };

ReactDOM.render(<DeepDiveArtifact run={run} view={view} template={template} />, root, () => {
    setTimeout(() => { window.__ddQA = preflightPages(root); window.__ddReady = true; }, 600);
});
