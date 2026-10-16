import { PackageRewrite } from './package-rewrite';
import * as React from 'react';
import { thesisFields, setText, revisionPackage, correctionBody, packageFields, applyPackageAlternative } from './package-editor-model.mjs';
import { claimLabel, editionDecision } from './onepager-edition-model.mjs';
export function PackageEditor({
  job,
  api
}) {
  var packageRef = job.result.reviewPackage,
    base = `/api/research/assignments/${job.id}`;
  var key = `charlie-package-editor:${api}:${job.id}:${packageRef.requestId}`;
  var [open, setOpen] = React.useState(false),
    [data, setData] = React.useState(null),
    [analysis, setAnalysis] = React.useState(null),
    [words, setWords] = React.useState({}),
    [tab, setTab] = React.useState('thesis'),
    [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState(''),
    [notice, setNotice] = React.useState(''),
    [prefs, setPrefs] = React.useState(null),
    [rule, setRule] = React.useState(''),
    [ack, setAck] = React.useState(false),
    [decisionAck, setDecisionAck] = React.useState(false);
  var attempt = React.useRef(null),
    locked = React.useRef(false),
    live = React.useRef(true);
  var dirty = !!data && (JSON.stringify(analysis) !== JSON.stringify(data.draft.package.analysis) || Object.entries(words).some(([id, v]) => v !== data.edition.record.claims.find(c => c.correctionId === id)?.proposal.statement));
  React.useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  React.useEffect(() => {
    if (!dirty && !attempt.current) return;
    var warn = e => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirty]);
  async function call(path, body) {
    var r = await fetch(api + path, {
      ...(body === undefined ? {} : {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      }),
      signal: AbortSignal.timeout(60000)
    });
    var d;
    try {
      d = await r.json();
    } catch {
      throw Error('The editor response was interrupted. Your saved package and edits are unchanged. Try opening the editor again.');
    }
    if (!r.ok) throw Error(d.error || 'Unable to save. Your edits are retained; retry the same save.');
    return d;
  }
  async function act(fn) {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setError('');
    try {
      await fn();
    } catch (e) {
      if (live.current) setError(e.message);
    } finally {
      locked.current = false;
      if (live.current) setBusy(false);
    }
  }
  async function load() {
    var [draft, pack] = await Promise.all([call('/api/thesis-imports/' + packageRef.thesisDraftId), call(base + '/onepager-editions')]);
    if (!Array.isArray(pack.editions) || !draft.package) throw Error('The editor response is incomplete. Your saved package is unchanged. Try opening the editor again.');
    var edition = pack.editions.find(e => e.id === packageRef.editionId);
    if (!edition || draft.id !== packageRef.thesisDraftId || draft.ticker !== job.ticker || pack.assignmentId !== job.id || pack.runId !== job.result.reportId) throw Error('The saved package could not be matched. Reload the assignment.');
    if (!live.current) return;
    var nextWords = Object.fromEntries(edition.record.claims.map(c => [c.correctionId, c.proposal.statement]));
    var stored = null;
    try {
      stored = JSON.parse(sessionStorage.getItem(key) || 'null');
    } catch {
      throw Error('Browser draft recovery is unavailable. Enable browser storage before editing.');
    }
    if (stored && (stored.fingerprint !== draft.fingerprint || stored.editionId !== edition.id)) throw Error('Saved edits belong to another version. Keep this tab and reload the matching package.');
    if (stored && (JSON.stringify(thesisFields(stored.analysis).map(f => f.path)) !== JSON.stringify(thesisFields(draft.package.analysis).map(f => f.path)) || Object.keys(nextWords).some(k => typeof stored.words?.[k] !== 'string'))) throw Error('Recovered edits could not be validated. Keep this tab open and copy your edits before proceeding.');
    setData({
      draft,
      pack,
      edition
    });
    setAnalysis(stored?.analysis || draft.package.analysis);
    setWords(stored?.words || nextWords);
    attempt.current = stored?.attempt || null;
    setOpen(true);
    if (stored) setNotice('Your unfinished edits were recovered. They have not been approved.');
    try {
      var p = await call('/api/thesis-imports/preferences');
      if (live.current) setPrefs(p);
    } catch {
      if (live.current) setNotice('Editing is available. Writing preferences could not be loaded.');
    }
  }
  function persist(a, w, saveAttempt = attempt.current) {
    var value = {
      fingerprint: data.draft.fingerprint,
      editionId: data.edition.id,
      analysis: a,
      words: w,
      attempt: saveAttempt
    };
    try {
      var encoded = JSON.stringify(value);
      sessionStorage.setItem(key, encoded);
      if (sessionStorage.getItem(key) !== encoded) throw Error();
    } catch {
      throw Error('Browser draft storage is unavailable. Keep this tab open and copy your edits before leaving.');
    }
  }
  function change(a, w) {
    setAnalysis(a);
    setWords(w);
    setAck(false);
    try {
      persist(a, w);
    } catch (e) {
      setError(e.message);
    }
  }
  async function save() {
    if (!ack) throw Error('Review the evidence and confirm your revised wording first.');
    if (data.draft.status !== 'pending' || data.draft.stale || data.edition.decision || data.edition.stale) throw Error('This version was decided or its baseline changed. Prepare a fresh review version.');
    if (!attempt.current) {
      var id = crypto.randomUUID();
      attempt.current = {
        id,
        thesis: revisionPackage(data.draft, analysis, id),
        corrections: data.edition.record.claims.filter(c => words[c.correctionId] !== c.proposal.statement).map(c => ({
          oldId: c.correctionId,
          body: correctionBody(c, words[c.correctionId], data.pack.fingerprint, crypto.randomUUID())
        })),
        editionId: crypto.randomUUID(),
        receipts: {}
      };
    }
    var a = attempt.current;
    persist(analysis, words, a);
    var draft = await call('/api/thesis-imports', a.thesis);
    a.draftId = draft.id;
    persist(analysis, words, a);
    for (var c of a.corrections) {
      if (!a.receipts[c.oldId]) {
        var r = await call(`/api/research/stock-analysis-run/${data.pack.runId}/corrections`, c.body);
        a.receipts[c.oldId] = r.id;
        persist(analysis, words, a);
      }
    }
    var eid = a.corrections.length ? a.editionId : data.edition.id;
    if (a.corrections.length) {
      await call(base + '/onepager-editions', {
        ...Object.fromEntries(['assignmentId', 'runId', 'fingerprint', 'fundamentalsDigest', 'parentId'].map(k => [k, data.pack[k]])),
        requestId: eid,
        correctionIds: data.edition.record.correctionIds.map(id => a.receipts[id] || id)
      });
    }
    // Validate the compact rendering before making it the visible package.
    await call(base + `/onepager-editions/${eid}/view?layout=recovery`);
    await call(base + '/review-package', {
      requestId: a.id,
      previousRequestId: packageRef.requestId,
      thesisDraftId: a.draftId,
      editionId: eid
    });
    sessionStorage.removeItem(key);
    attempt.current = null;
    setNotice('New package version saved. Reloading the matching previews…');
    window.location.reload();
  }
  function discard() {
    if (attempt.current || !window.confirm('Discard the unsaved wording edits in this tab? Saved versions will remain unchanged.')) return;
    sessionStorage.removeItem(key);
    setAnalysis(data.draft.package.analysis);
    setWords(Object.fromEntries(data.edition.record.claims.map(c => [c.correctionId, c.proposal.statement])));
    setAck(false);
    setNotice('Unsaved edits discarded. The saved package is unchanged.');
  }
  async function preference(rules) {
    var value = await call('/api/thesis-imports/preferences', {
      rules,
      fingerprint: prefs.fingerprint
    });
    if (live.current) {
      setPrefs(value);
      setRule('');
      setNotice('Writing preferences saved for future stock research and thesis preparation. Existing drafts are unchanged.');
    }
  }
  async function decide(action) {
    var body = editionDecision(data.edition, data.pack, action, crypto.randomUUID());
    await call(base + `/onepager-editions/${data.edition.id}/decision`, body);
    setDecisionAck(false);
    setNotice('One-pager decision saved. Thesis approval is separate.');
    await load();
  }
  return /*#__PURE__*/React.createElement("section", {
    className: "package-editor"
  }, /*#__PURE__*/React.createElement("style", null, `.package-editor{padding:20px 24px;border-bottom:1px solid #ccc;font-family:Calibri,sans-serif;color:#000}.package-editor *{box-sizing:border-box;color:#000}.package-editor h3{margin:0 0 8px}.package-editor textarea{display:block;width:100%;font:16px/1.5 Calibri,sans-serif;padding:12px;border:1px solid #aaa;border-radius:4px;background:#fff;resize:vertical}.package-editor label{display:block;margin:14px 0}.package-editor fieldset{border:0;padding:0;min-width:0}.package-editor article{padding:14px 0;border-top:1px solid #ddd}.package-editor .pe-actions{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.package-editor button{min-height:42px}.package-editor .pe-note{padding:12px;background:#f2f1eb;line-height:1.5}.package-editor summary{cursor:pointer}.package-editor blockquote{white-space:pre-wrap;overflow-wrap:anywhere}.package-editor a{color:#000!important;text-decoration:underline}@media(max-width:650px){.package-editor{padding:16px}.package-editor .pe-actions>button{flex:1}}`), dirty && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Unsaved wording changes \xB7 saved locally in this browser tab."), /*#__PURE__*/React.createElement("h3", null, "Your research, your wording"), /*#__PURE__*/React.createElement("p", null, "Edit the draft here, save a new version, then review before approving. Earlier versions stay preserved."), !open ? /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(load)
  }, busy ? 'Opening editor…' : 'Edit & review this package') : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "pe-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setTab('thesis')
  }, "Stock summary & thesis"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setTab('brief')
  }, "One-pager"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setTab('preferences')
  }, "Writing preferences"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setOpen(false)
  }, "Close editor \xB7 keep edits")), tab === 'preferences' ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, "What should Charlie remember?"), /*#__PURE__*/React.createElement("p", null, "Save reusable writing rules explicitly. Stock-specific forecasts and investment judgments stay in the draft. These rules guide future stock research and thesis preparation; this is preference memory, not model training."), prefs ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("ul", null, prefs.rules.map((r, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, r, " ", /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(() => preference(prefs.rules.filter((_, n) => n !== i)))
  }, "Remove")))), /*#__PURE__*/React.createElement("label", null, "Remember for future writing", /*#__PURE__*/React.createElement("textarea", {
    maxLength: 500,
    rows: 3,
    value: rule,
    onChange: e => setRule(e.target.value),
    placeholder: "For example: keep broker names in source references, never in the narrative."
  })), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !rule.trim() || prefs.rules.length >= 20,
    onClick: () => act(() => preference([...prefs.rules, rule.trim()]))
  }, "Remember this preference")) : /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(async () => setPrefs(await call('/api/thesis-imports/preferences')))
  }, "Load writing preferences")) : /*#__PURE__*/React.createElement(React.Fragment, null, (data.draft.status !== 'pending' || data.draft.stale || data.edition.decision || data.edition.stale) && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "This saved version has been decided or its baseline has changed. Editing is disabled to preserve that decision; a fresh research revision is required."), /*#__PURE__*/React.createElement("p", {
    className: "pe-note"
  }, tab === 'thesis' ? 'The stock summary and proposed thesis share these sections. Saving updates both. Review the one-pager separately: concise wording is never silently rewritten.' : 'Edit the one-pager’s saved claims. Evidence references remain attached; changing wording does not prove the new claim. Keep each field concise.'), /*#__PURE__*/React.createElement(PackageRewrite, {
    key: tab,
    runId: job.result.reportId,
    storageKey: key + ':' + tab,
    api: api,
    disabled: busy || !!attempt.current || data.draft.status !== 'pending' || data.draft.stale || !!data.edition.decision || data.edition.stale,
    fields: packageFields(analysis, data.edition.record.claims, words),
    onKeep: values => {
      var next = applyPackageAlternative(analysis, words, values);
      change(next.analysis, next.words);
      setNotice('Package alternative kept in your draft. Review both documents and their evidence before saving.');
    }
  }), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || !!attempt.current || data?.draft.status !== 'pending' || data?.draft.stale || data?.edition.decision || data?.edition.stale
  }, tab === 'thesis' ? thesisFields(analysis).map(f => /*#__PURE__*/React.createElement("label", {
    key: JSON.stringify(f.path)
  }, f.label, /*#__PURE__*/React.createElement("textarea", {
    rows: f.path.at(-1) === 'summary' ? 8 : 3,
    maxLength: 40000,
    value: f.value,
    onChange: e => change(setText(analysis, f.path, e.target.value), words)
  }))) : data.edition.record.claims.map(c => /*#__PURE__*/React.createElement("article", {
    key: c.correctionId
  }, /*#__PURE__*/React.createElement("label", null, claimLabel(c.proposal.path), /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    maxLength: 1600,
    value: words[c.correctionId],
    onChange: e => change(analysis, {
      ...words,
      [c.correctionId]: e.target.value
    })
  })), /*#__PURE__*/React.createElement("small", null, words[c.correctionId].length, "/1600 characters"), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Original wording & evidence"), /*#__PURE__*/React.createElement("p", null, c.proposal.statement), [c.proposal, ...(c.proposal.additionalEvidence || [])].map((e, i) => /*#__PURE__*/React.createElement("div", {
    key: i
  }, /*#__PURE__*/React.createElement("p", null, e.locator), /*#__PURE__*/React.createElement("blockquote", null, e.excerpt))))))), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: ack,
    disabled: busy,
    onChange: e => setAck(e.target.checked)
  }), " I checked my revised wording against its evidence and reviewed both the summary/thesis and the one-pager."), /*#__PURE__*/React.createElement("div", {
    className: "pe-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy || !ack || !dirty && !attempt.current,
    onClick: () => act(save)
  }, attempt.current ? 'Retry saving this exact revision' : 'Save new package version'), dirty && !attempt.current && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: discard
  }, "Discard unsaved edits")), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Thesis evidence and source limitations"), /*#__PURE__*/React.createElement("p", null, (data.draft.package.provenance?.limitations || []).join(' ')), data.draft.package.sourceRegister.map(s => /*#__PURE__*/React.createElement("p", {
    key: s.id
  }, s.id, " \xB7 ", s.filename || s.name, " \xB7 pages ", (s.reviewedPdfPages || []).join(', ')))), /*#__PURE__*/React.createElement("p", null, "Saving does not approve research. Unfinished edits are retained in this browser tab; save a package version to retain them across devices."), !dirty && !attempt.current && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Decide on this saved version"), /*#__PURE__*/React.createElement("p", null, "Thesis approval changes the saved detailed thesis. One-pager acceptance records a separate editorial decision."), /*#__PURE__*/React.createElement("a", {
    href: `?thesisDraft=${packageRef.thesisDraftId}#view=thesisimports&ticker=${job.ticker}`
  }, "Compare and approve or dismiss the thesis"), data.pack.canDecide && !data.edition.decision && !data.edition.stale && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: decisionAck,
    onChange: e => setDecisionAck(e.target.checked)
  }), " I reviewed this exact one-pager\u2019s wording, evidence and scenario assumptions."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !decisionAck,
    onClick: () => act(() => decide('accept'))
  }, "Accept this one-pager"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !decisionAck,
    onClick: () => act(() => decide('reject'))
  }, "Reject this one-pager")), !data.pack.canDecide && /*#__PURE__*/React.createElement("p", null, "Sign in as the investor to record a one-pager decision."), data.edition.decision && /*#__PURE__*/React.createElement("p", null, "One-pager decision: ", data.edition.decision.action, ".")))), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), notice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice));
}