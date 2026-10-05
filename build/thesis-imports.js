import React, { useState, useEffect } from 'react';
import { parseDraft, canApprove, downloadFile, MAX_DRAFT_BYTES } from './thesis-import-model.mjs';
var labels = {
  pdfPages: 'PDF pages',
  sourceId: 'Source',
  sha256: 'File hash',
  claimBasis: 'Basis',
  thresholdBasis: 'Threshold basis',
  reviewTrigger: 'Review trigger',
  triggerPoints: 'Trigger points',
  sourceRegister: 'Source register',
  supportType: 'Support',
  usageStatus: 'Source-use note',
  reviewStatus: 'Review coverage'
};
var label = key => labels[key] || key.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());
function Content({
  value
}) {
  if (value == null || value === '') return /*#__PURE__*/React.createElement("span", {
    className: "ti-muted"
  }, "Not supplied");
  if (typeof value !== 'object') return /*#__PURE__*/React.createElement("span", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, String(value));
  if (Array.isArray(value)) return /*#__PURE__*/React.createElement("div", {
    className: "ti-items"
  }, value.length ? value.map((v, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    className: typeof v === 'object' ? 'ti-item' : ''
  }, /*#__PURE__*/React.createElement(Content, {
    value: v
  }))) : /*#__PURE__*/React.createElement("span", {
    className: "ti-muted"
  }, "None"));
  return /*#__PURE__*/React.createElement("dl", null, Object.entries(value).filter(([k, v]) => k !== 'id' && v !== '' && v != null).map(([k, v]) => /*#__PURE__*/React.createElement("div", {
    key: k,
    className: "ti-field"
  }, k === 'sources' ? /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source references (", Array.isArray(v) ? v.length : 0, ")"), /*#__PURE__*/React.createElement(Content, {
    value: v
  })) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("dt", null, label(k)), /*#__PURE__*/React.createElement("dd", null, /*#__PURE__*/React.createElement(Content, {
    value: v
  }))))));
}
export function ThesisImports({
  api = '',
  initialTicker = '',
  onNavigate,
  onSaved
}) {
  var [ticker, setTicker] = useState(initialTicker);
  var [drafts, setDrafts] = useState([]),
    [draft, setDraft] = useState(null);
  var [busy, setBusy] = useState(false),
    [error, setError] = useState(''),
    [notice, setNotice] = useState('');
  var [checked, setChecked] = useState(false),
    [confirmation, setConfirmation] = useState('');
  var [loaded, setLoaded] = useState(false),
    [filter, setFilter] = useState('');
  async function call(path, body) {
    var r = await fetch(`${api}/api/thesis-imports${path}`, body === undefined ? {} : {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });
    var d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `Request failed (${r.status}). Your draft has not been discarded; refresh to check its status before retrying.`);
    return d;
  }
  async function list() {
    var data = await call(filter ? `?ticker=${encodeURIComponent(filter)}` : '');
    setDrafts(data.drafts);
    setLoaded(true);
  }
  async function select(id) {
    var d = await call(`/${id}`);
    setDraft(d);
    setTicker(d.ticker);
    setChecked(false);
    setConfirmation('');
  }
  async function act(fn) {
    setBusy(true);
    setError('');
    setNotice('');
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    var active = true;
    call('').then(d => {
      if (active) {
        setDrafts(d.drafts);
        setLoaded(true);
      }
    }).catch(e => {
      if (active) setError(e.message);
    });
    return () => {
      active = false;
    };
  }, [api]);
  async function prepare() {
    if (!/^[A-Z0-9][A-Z0-9.-]{0,19}$/.test(ticker)) throw new Error('Enter a ticker before preparing a thesis.');
    var data = await call(`/prepare/${encodeURIComponent(ticker)}`);
    downloadFile(`${ticker}-prepare-for-ChatGPT.json`, {
      ...data.package,
      instructions: data.instructions
    });
    setNotice('Preparation file downloaded. Attach it and your chosen source documents in ChatGPT. Ask for the completed JSON draft, then import that file here.');
  }
  async function upload(file) {
    if (!file) return;
    if (file.size > MAX_DRAFT_BYTES) throw new Error('Choose a draft smaller than 2 MB.');
    var p = parseDraft(await file.text());
    var d = await call('', p);
    await select(d.id);
    await list();
    setNotice(d.replayed ? 'This file is already in your inbox. Its saved review is open below.' : 'Draft saved for review. The live thesis has not changed.');
  }
  async function approve() {
    await call(`/${draft.id}/approve`, {
      confirm: true,
      ticker: confirmation,
      fingerprint: draft.fingerprint
    });
    await select(draft.id);
    await list();
    setNotice('Thesis approved and saved, with source references and a restorable revision. No model calls were made.');
    onSaved?.();
  }
  return /*#__PURE__*/React.createElement("div", {
    className: "ti-workspace"
  }, /*#__PURE__*/React.createElement("style", null, `
      .ti-workspace{overflow:auto;padding:24px 24px 100px;flex:1;min-width:0;color:#000;background:#f4f3ef;font:16px/1.5 Calibri,Carlito,sans-serif}
      .ti-workspace *{box-sizing:border-box;font-family:Calibri,Carlito,sans-serif!important;color:#000!important}.ti-workspace input::placeholder{color:#000;opacity:.65}.ti-inner{max-width:1260px;margin:auto}.ti-workspace h1{font-size:30px;font-weight:700;margin:0}.ti-workspace h2{font-size:21px;font-weight:700;margin:0 0 10px}.ti-workspace h3{font-weight:700}
      .ti-workspace p{margin:8px 0}.ti-muted{font-size:14px}.ti-kicker{font-size:12px;letter-spacing:.13em;text-transform:uppercase}.ti-actions{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:16px 0}.ti-workspace button,.ti-upload{border:1px solid #aaa;border-radius:7px;padding:9px 14px;background:#fff;color:#000;cursor:pointer;font-weight:600}
      .ti-workspace button:disabled{opacity:.5;cursor:wait}.ti-workspace button:focus-visible,.ti-workspace input:focus-visible,.ti-upload:focus-within{outline:3px solid #647c8a;outline-offset:2px}.ti-workspace input[type=text]{min-width:0;border:1px solid #999;border-radius:5px;padding:9px;background:white;color:#000}.ti-workspace .ti-primary{background:#dbe8d5;border-color:#7a8c70}.ti-upload input{position:absolute;opacity:0;width:1px;height:1px}.ti-panel{padding:22px;background:#fff;border:1px solid #d8d5cc;border-radius:10px;margin-top:20px}.ti-message{padding:13px 16px;border:1px solid #b0baaa;background:#eaf1e5;border-radius:7px;margin:15px 0;overflow-wrap:anywhere}.ti-error{background:#fbe7df;border-color:#c28d77}
      .ti-inbox{display:flex;gap:8px;overflow:auto;padding:5px 0 10px}.ti-inbox button{text-align:left;flex-shrink:0;max-width:270px;white-space:normal}.ti-inbox button[aria-pressed=true]{border:2px solid #485744;background:#eaf1e5}.ti-status{font-size:12px;text-transform:uppercase;letter-spacing:.06em}.ti-section{border:1px solid #d8d5cc;border-radius:8px;overflow:hidden;margin:18px 0}.ti-section-head{padding:12px 16px;background:#eeece5;display:flex;justify-content:space-between;gap:12px}.ti-comparison{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.ti-column{padding:18px;min-width:0;overflow-wrap:anywhere}.ti-column+.ti-column{border-left:1px solid #ddd}.ti-column-title{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:14px}
      .ti-field{margin:8px 0}.ti-field dt{font-size:13px;font-weight:700}.ti-field dd{margin:2px 0 10px}.ti-item{border-top:1px solid #ddd;padding:10px 0}.ti-workspace details{margin:12px 0;overflow-wrap:anywhere}.ti-workspace summary{cursor:pointer;font-weight:600}.ti-workspace dl{margin:0}.ti-approval{background:#f1f4ed}.ti-check{display:flex;align-items:flex-start;gap:10px}.ti-check input{margin-top:6px;flex-shrink:0}.ti-workspace ul{list-style:disc;padding-left:22px}.ti-guide{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin:22px 0}.ti-guide strong{display:block}.ti-busy{font-size:14px}
      @media(max-width:700px){.ti-workspace{padding:16px 12px 110px}.ti-guide{grid-template-columns:1fr;gap:12px}.ti-panel{padding:15px}.ti-comparison{grid-template-columns:1fr}.ti-column+.ti-column{border-left:0;border-top:1px solid #ddd}.ti-workspace h1{font-size:26px}.ti-actions{align-items:stretch}.ti-actions input[type=text]{width:100%}.ti-actions button,.ti-upload{text-align:center;max-width:100%}}
    `), /*#__PURE__*/React.createElement("div", {
    className: "ti-inner"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ti-kicker"
  }, "Investment thesis \xB7 External drafts"), /*#__PURE__*/React.createElement("h1", null, "Bring your research into Charlie"), /*#__PURE__*/React.createElement("p", null, "Write with your subscription. Review here. Keep the same thesis history and future upgrade workflow."), /*#__PURE__*/React.createElement("div", {
    className: "ti-guide"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "01 \xB7 Prepare for ChatGPT"), /*#__PURE__*/React.createElement("span", null, "Download a template with the current thesis and its starting version. Attach your own sources in ChatGPT.")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "02 \xB7 Import draft"), /*#__PURE__*/React.createElement("span", null, "Upload the completed JSON file. Drafts stay in this inbox until you approve or dismiss them.")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "03 \xB7 Review changes"), /*#__PURE__*/React.createElement("span", null, "Compare every section and source reference, then approve the new thesis or upgrade."))), /*#__PURE__*/React.createElement("div", {
    className: "ti-actions"
  }, /*#__PURE__*/React.createElement("label", null, "Ticker ", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Preparation ticker",
    type: "text",
    value: ticker,
    maxLength: 20,
    onChange: e => setTicker(e.target.value.toUpperCase().trim()),
    disabled: busy
  })), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(prepare)
  }, "Prepare for ChatGPT"), /*#__PURE__*/React.createElement("label", {
    className: "ti-upload"
  }, "Import draft", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Import draft file",
    type: "file",
    accept: ".json,application/json",
    disabled: busy,
    onChange: e => {
      var file = e.target.files[0];
      e.target.value = '';
      act(() => upload(file));
    }
  })), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(async () => {
      await list();
      if (draft) await select(draft.id);
    })
  }, "Refresh inbox")), /*#__PURE__*/React.createElement("p", {
    className: "ti-muted"
  }, "Preparation, import and approval make no paid model calls. Original documents stay where you keep them; references do not upload or verify the originals."), busy && /*#__PURE__*/React.createElement("p", {
    className: "ti-busy",
    role: "status"
  }, "Working\u2026"), error && /*#__PURE__*/React.createElement("div", {
    role: "alert",
    className: "ti-message ti-error"
  }, error), notice && /*#__PURE__*/React.createElement("div", {
    role: "status",
    className: "ti-message"
  }, notice), /*#__PURE__*/React.createElement("section", {
    className: "ti-panel"
  }, /*#__PURE__*/React.createElement("h2", null, "Draft inbox"), /*#__PURE__*/React.createElement("p", {
    className: "ti-muted"
  }, "Most recent 100 matching drafts. Reimporting the same file opens its existing review."), /*#__PURE__*/React.createElement("div", {
    className: "ti-actions"
  }, /*#__PURE__*/React.createElement("label", null, "Filter by ticker ", /*#__PURE__*/React.createElement("input", {
    type: "text",
    "aria-label": "Inbox ticker filter",
    placeholder: "All companies",
    maxLength: 20,
    value: filter,
    disabled: busy,
    onChange: e => setFilter(e.target.value.toUpperCase().trim())
  })), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(list)
  }, "Find drafts")), loaded && !drafts.length && /*#__PURE__*/React.createElement("p", null, "No drafts yet for this view. Import a structured draft or prepare a company above."), /*#__PURE__*/React.createElement("div", {
    className: "ti-inbox"
  }, drafts.map(d => /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    key: d.id,
    "aria-pressed": draft?.id === d.id,
    onClick: () => act(() => select(d.id))
  }, /*#__PURE__*/React.createElement("strong", null, d.ticker, " \xB7 ", d.company), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    className: "ti-status"
  }, d.status), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    className: "ti-muted"
  }, new Date(d.created_at).toLocaleString()))))), draft && /*#__PURE__*/React.createElement("section", {
    className: "ti-panel",
    "aria-label": "Draft review"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ti-kicker"
  }, draft.baseline ? 'Thesis upgrade' : 'Initial thesis', " \xB7 ", draft.status), /*#__PURE__*/React.createElement("h2", null, draft.ticker, " \u2014 ", draft.package.companyName), /*#__PURE__*/React.createElement("p", null, draft.sections.filter(s => s.changed).length, " of 4 sections changed. Read both versions, including removed items."), /*#__PURE__*/React.createElement("div", {
    className: "ti-message"
  }, /*#__PURE__*/React.createElement("ul", null, draft.warnings.map((w, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, w)))), draft.stale && /*#__PURE__*/React.createElement("div", {
    className: "ti-message ti-error",
    role: "alert"
  }, "The live thesis has changed. Approval is blocked. Download a fresh preparation file, reconcile this draft in ChatGPT, and import the revised result."), /*#__PURE__*/React.createElement("div", {
    className: "ti-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => downloadFile(`${draft.ticker}-draft.json`, draft.package)
  }, "Download this draft"), draft.status === 'approved' && /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('portfolio', draft.ticker)
  }, "Open saved thesis")), draft.sections.map(s => /*#__PURE__*/React.createElement("section", {
    key: s.key,
    className: "ti-section"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ti-section-head"
  }, /*#__PURE__*/React.createElement("h3", null, s.name), /*#__PURE__*/React.createElement("span", null, s.changed ? 'Changed' : 'Unchanged')), /*#__PURE__*/React.createElement("div", {
    className: "ti-comparison"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ti-column"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ti-column-title"
  }, "Before \xB7 saved at import"), /*#__PURE__*/React.createElement(Content, {
    value: s.before
  })), /*#__PURE__*/React.createElement("div", {
    className: "ti-column"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ti-column-title"
  }, "Proposed"), /*#__PURE__*/React.createElement(Content, {
    value: s.after
  }))))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source register (", draft.package.sourceRegister.length, ") \xB7 author-supplied metadata"), /*#__PURE__*/React.createElement(Content, {
    value: draft.package.sourceRegister
  })), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Authorship and preparation details \xB7 author supplied"), /*#__PURE__*/React.createElement(Content, {
    value: draft.package.provenance
  })), draft.status === 'pending' && /*#__PURE__*/React.createElement("section", {
    className: "ti-panel ti-approval"
  }, /*#__PURE__*/React.createElement("h2", null, "Your approval"), /*#__PURE__*/React.createElement("p", null, "This will ", draft.baseline ? 'replace the current detailed thesis' : 'create the initial thesis', " for ", draft.ticker, ". The reviewed draft is preserved, and upgrades retain the previous thesis for restoration."), /*#__PURE__*/React.createElement("label", {
    className: "ti-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: checked,
    disabled: busy || draft.stale,
    onChange: e => setChecked(e.target.checked)
  }), /*#__PURE__*/React.createElement("span", null, "I reviewed the proposed thesis, signposts, risks and source limitations, and approve saving this version.")), /*#__PURE__*/React.createElement("div", {
    className: "ti-actions"
  }, /*#__PURE__*/React.createElement("label", null, "Type ", draft.ticker, " to confirm ", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Confirm ticker",
    type: "text",
    value: confirmation,
    disabled: busy || draft.stale,
    onChange: e => setConfirmation(e.target.value.toUpperCase().trim())
  })), /*#__PURE__*/React.createElement("button", {
    className: "ti-primary",
    disabled: !canApprove(draft, checked, confirmation, busy),
    onClick: () => act(approve)
  }, "Approve & save thesis"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(async () => {
      await call(`/${draft.id}/dismiss`, {});
      await select(draft.id);
      await list();
    })
  }, "Dismiss draft")), /*#__PURE__*/React.createElement("p", {
    className: "ti-muted"
  }, "For edits, download this draft, revise it in ChatGPT, then import the new file. Paid research and report generation remain separate choices.")))));
}