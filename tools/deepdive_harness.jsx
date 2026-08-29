// Deterministic golden-render harness.
//   ?view=onepager|twopager|memo   ?fixture=de|unh   ?template=notebook|...
// UNH is the hard case: a long company name, four overlapping segments whose
// shares sum to 119%, and no logo asset -- exactly what a universal template
// has to survive.
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { DeepDiveArtifact, preflightPages } from '../src/deepdive';
import de from '../fixtures/deepdive_de_golden.json';
import unh from '../fixtures/deepdive_unh_sample.json';

const q = new URLSearchParams(location.search);
const src = q.get('fixture') === 'unh' ? unh : de;
const root = document.getElementById('root');
const run = { master: src.master, onepager: src.onepager };

ReactDOM.render(
    <DeepDiveArtifact run={run} view={q.get('view') || 'onepager'}
                      template={q.get('template') || 'notebook'} />,
    root,
    () => {
        // Measure only after the rebalancer's last pass (1200ms) has landed.
        const measure = () => { window.__ddQA = preflightPages(root); window.__ddReady = true; };
        window.addEventListener('deepdive:layout-settled', measure);
        setTimeout(measure, 1800);
    });
