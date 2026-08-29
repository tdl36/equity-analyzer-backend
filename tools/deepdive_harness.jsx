// Deterministic golden-render harness.
//   ?view=onepager|twopager|memo  ?fixture=de|unh  ?template=...  ?chrome=1
//
// ?chrome=1 wraps the artifact in decoy nesting that mimics Charlie's real
// layout -- scrolling containers, toolbars, a fit wrapper. Without it the print
// path tests nothing useful, because the bug being chased is precisely what the
// app's chrome does to the printed page.
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { DeepDiveArtifact, PageFit, preflightPages, printArtifact } from '../src/deepdive';
import de from '../fixtures/deepdive_de_golden.json';
import unh from '../fixtures/deepdive_unh_sample.json';

const q = new URLSearchParams(location.search);
const src = q.get('fixture') === 'unh' ? unh : de;
const root = document.getElementById('root');
const run = { master: src.master, onepager: src.onepager };
const view = q.get('view') || 'onepager';

const artifact = <DeepDiveArtifact run={run} view={view} template={q.get('template') || 'notebook'} />;

const tree = q.get('chrome') === '1'
    ? (
        <div style={{ height: '100vh', overflowY: 'auto', background: '#14110c' }}>
            <div style={{ padding: 24 }}>
                <div style={{ height: 60, background: '#222' }}>decoy nav</div>
                <div style={{ height: 90, background: '#333' }}>decoy toolbar</div>
                <div className="dd-stage">
                    <PageFit>{artifact}</PageFit>
                </div>
            </div>
        </div>
      )
    : artifact;

ReactDOM.render(tree, root, () => {
    const measure = () => { window.__ddQA = preflightPages(root); window.__ddReady = true; };
    window.addEventListener('deepdive:layout-settled', measure);
    setTimeout(measure, 1800);
    // Exposed so a headless run can exercise the real print path.
    window.__ddPrint = () => printArtifact(view);
});
