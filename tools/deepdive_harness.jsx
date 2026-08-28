// Deterministic golden-render harness for the Deep Dive artifacts.
//
// Renders the three artifacts from the reviewed Deere fixture with no backend,
// no auth and no app chrome, so layout can be inspected and screenshotted
// directly. This is the fixture the handoff's visual-QA plan calls for: the
// only way to compare against best_proven_outputs/DE_v29_*.pdf is to look at
// the same data rendered by this code.
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { DeepDiveOnePager, DeepDiveTwoPager, DeepDiveMemo, preflightPages } from '../src/deepdive';
import fixture from '../fixtures/deepdive_de_golden.json';

const view = new URLSearchParams(location.search).get('view') || 'onepager';
const root = document.getElementById('root');

const node =
    view === 'twopager' ? <DeepDiveTwoPager master={fixture.master} /> :
    view === 'memo'     ? <DeepDiveMemo master={fixture.master} /> :
                          <DeepDiveOnePager data={fixture.onepager} />;

ReactDOM.render(node, root, () => {
    // Publish preflight for the screenshot script to assert on.
    setTimeout(() => {
        window.__ddQA = preflightPages(root);
        window.__ddReady = true;
    }, 400);
});
