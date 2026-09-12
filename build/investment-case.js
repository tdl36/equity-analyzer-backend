import * as React from 'react';
import { InvestorFramework } from './investor-framework';
import { ResearchDecisions } from './research-decisions';
import { ResearchWorkbench } from './research-workbench';
import { UnderweightMonitor } from './underweight-monitor';
import { ThesisEvolution } from './thesis-evolution';
import { CaseSignals } from './case-signals';
import { CaseEvidence } from './investment-case-evidence';
var empty = () => ({
  thesis: '',
  variantView: '',
  marketBaseline: '',
  changeConditions: '',
  assumptions: [],
  scenarios: {}
});
var fields = [['thesis', 'Investment thesis'], ['variantView', 'Where my view differs'], ['marketBaseline', 'Market expectations · include source and date'], ['changeConditions', 'What would change my mind']];
export function InvestmentCase({
  api,
  analyses = [],
  initialTicker = ''
}) {
  var [ticker, setTicker] = React.useState(initialTicker),
    [active, setActive] = React.useState(''),
    [body, setBody] = React.useState(empty),
    [revision, setRevision] = React.useState(0),
    [versions, setVersions] = React.useState([]),
    [bridge, setBridge] = React.useState({}),
    [busy, setBusy] = React.useState(false),
    [message, setMessage] = React.useState(''),
    [dirty, setDirty] = React.useState(false),
    [selectedVersion, setSelectedVersion] = React.useState('');
  var [workspaceTab, setWorkspaceTab] = React.useState('case');
  var pending = React.useRef(null),
    lock = React.useRef(false),
    alive = React.useRef(true);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  var json = async (path, options = {}) => {
    var r = await fetch(api + path, {
      ...options,
      signal: AbortSignal.timeout(20000)
    });
    var d = await r.json();
    if (!r.ok) throw Error(d.error || 'Investment case unavailable');
    return d;
  };
  var load = async (selectedTicker = ticker) => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setMessage('');
    try {
      var t = (typeof selectedTicker === 'string' ? selectedTicker : ticker).trim().toUpperCase();
      setTicker(t);
      var d = await json('/api/research/investment-case/' + encodeURIComponent(t));
      if (!alive.current) return;
      setActive(t);
      setRevision(d.revision);
      setBody({
        ...empty(),
        ...d.body
      });
      setVersions(d.versions);
      setBridge(d.bridge);
      setDirty(false);
      setSelectedVersion('');
      pending.current = null;
    } catch (e) {
      if (alive.current) setMessage(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  React.useEffect(() => {
    if (initialTicker) load();
  }, []);
  var edit = next => {
    setBody(next);
    setDirty(true);
    pending.current = null;
    setMessage('Unsaved changes');
  };
  var save = async (operation = {
    body
  }) => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setMessage('Saving…');
    var signature = JSON.stringify({
      revision,
      ...operation
    });
    var payload = pending.current?.signature === signature ? pending.current.payload : {
      requestId: crypto.randomUUID(),
      revision,
      ...operation
    };
    pending.current = {
      signature,
      payload
    };
    try {
      var d = await json('/api/research/investment-case/' + encodeURIComponent(active), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      if (!alive.current) return;
      setRevision(d.revision);
      setBody(d.body || payload.body);
      setBridge(d.bridge || {});
      setDirty(false);
      pending.current = null;
      setMessage(`Saved revision ${d.revision}. Your thesis documents remain separate.`);
      try {
        var fresh = await json('/api/research/investment-case/' + encodeURIComponent(active));
        if (alive.current) {
          setVersions(fresh.versions);
        }
      } catch {
        if (alive.current) setMessage(`Revision ${d.revision} saved; history refresh unavailable.`);
      }
    } catch (e) {
      if (alive.current) setMessage(`${e.message} Your edits remain here. Retrying unchanged edits uses the same save request.`);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var assumption = (i, key, value) => edit({
    ...body,
    assumptions: body.assumptions.map((a, j) => i === j ? {
      ...a,
      [key]: value
    } : a)
  });
  var snapshot = versions.find(v => String(v.revision) === selectedVersion);
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel",
    "aria-label": "Investment case"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "INVESTMENT CASE / YOUR ASSUMPTIONS"), /*#__PURE__*/React.createElement("h2", null, "Investment thesis workspace."), /*#__PURE__*/React.createElement("p", null, "Record what you believe, the evidence against it, and the next test. These are your working assumptions. Manual source references are unverified; accepted research links retain the original excerpt and its provenance."), /*#__PURE__*/React.createElement(InvestorFramework, {
    api: api
  }), /*#__PURE__*/React.createElement(UnderweightMonitor, {
    api: api,
    disabled: busy || dirty,
    onOpen: t => {
      setWorkspaceTab('reviews');
      load(t);
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Company", /*#__PURE__*/React.createElement("input", {
    list: "investment-case-tickers",
    value: ticker,
    disabled: busy || dirty,
    onChange: e => setTicker(e.target.value.toUpperCase()),
    placeholder: "ABBV",
    maxLength: 20
  })), /*#__PURE__*/React.createElement("datalist", {
    id: "investment-case-tickers"
  }, [...new Set(analyses.map(a => a.ticker).filter(Boolean))].sort().map(t => /*#__PURE__*/React.createElement("option", {
    key: t,
    value: t
  }))), /*#__PURE__*/React.createElement("button", {
    disabled: busy || dirty || !ticker.trim(),
    onClick: load
  }, "Open investment case")), dirty && /*#__PURE__*/React.createElement("p", null, "Save your edits before switching companies. To discard them, use ", /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => {
      setDirty(false);
      setMessage('Edits remain visible until you open a company again.');
    }
  }, "Allow reload without saving"), "."), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), active && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h3", null, active, " \xB7 ", revision ? `Revision ${revision}` : 'New investment case'), /*#__PURE__*/React.createElement("span", null, dirty ? 'Unsaved changes' : 'Saved working assumptions')), /*#__PURE__*/React.createElement("nav", {
    className: "lifecycle-tabs",
    "aria-label": "Investment thesis workspace"
  }, [['case', 'Current thesis'], ['evidence', 'Evidence & proposals'], ['reviews', 'Decisions & underweights'], ['signals', 'Case signals'], ['evolution', 'Evolution']].map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    "aria-pressed": workspaceTab === id,
    onClick: () => setWorkspaceTab(id)
  }, label))), /*#__PURE__*/React.createElement("div", {
    hidden: workspaceTab !== 'case'
  }, !revision && /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h3", null, "Start from your saved thesis"), /*#__PURE__*/React.createElement("p", null, "Copy its summary, pillars and textual risks/signposts into an unsaved baseline for review. Original thesis documents remain unchanged."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || dirty,
    onClick: async () => {
      if (lock.current) return;
      lock.current = true;
      setBusy(true);
      try {
        var d = await json('/api/research/investment-case/' + active + '/saved-thesis-draft');
        if (alive.current) {
          edit(d.body);
          setMessage(d.scope + ' Review the draft below, then save the first revision.');
        }
      } catch (e) {
        if (alive.current) setMessage(e.message);
      } finally {
        lock.current = false;
        if (alive.current) setBusy(false);
      }
    }
  }, "Load saved thesis as a draft")), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy,
    className: "desk-form"
  }, fields.map(([key, label]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    value: body[key],
    onChange: e => edit({
      ...body,
      [key]: e.target.value
    })
  }))), /*#__PURE__*/React.createElement("h3", null, "Key assumptions and evidence"), body.assumptions.map((a, i) => /*#__PURE__*/React.createElement("section", {
    className: "amendment-card",
    key: a.id
  }, /*#__PURE__*/React.createElement("label", null, "Assumption ", i + 1, /*#__PURE__*/React.createElement("textarea", {
    rows: 2,
    value: a.claim,
    onChange: e => assumption(i, 'claim', e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Basis", /*#__PURE__*/React.createElement("select", {
    value: a.evidenceType,
    onChange: e => assumption(i, 'evidenceType', e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "interpretation"
  }, "My interpretation / assumption"), /*#__PURE__*/React.createElement("option", {
    value: "management_statement"
  }, "Management statement \xB7 attributed"), /*#__PURE__*/React.createElement("option", {
    value: "reported_fact"
  }, "Reported fact \xB7 verify against source"), /*#__PURE__*/React.createElement("option", {
    value: "broker_estimate"
  }, "Broker estimate \xB7 attributed"))), [['support', 'Supporting evidence'], ['contrary', 'Contrary evidence / unresolved'], ['nextTest', 'Next test and timing'], ['sourceReference', 'Source reference · document, page, date']].map(([key, label]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("textarea", {
    rows: 2,
    value: a[key],
    onChange: e => assumption(i, key, e.target.value)
  }))), /*#__PURE__*/React.createElement("button", {
    onClick: () => edit({
      ...body,
      assumptions: body.assumptions.filter((_, j) => j !== i)
    })
  }, "Remove assumption from this revision"))), /*#__PURE__*/React.createElement("button", {
    disabled: body.assumptions.length >= 30,
    onClick: () => edit({
      ...body,
      assumptions: [...body.assumptions, {
        id: crypto.randomUUID(),
        claim: '',
        support: '',
        contrary: '',
        nextTest: '',
        sourceReference: '',
        evidenceType: 'interpretation'
      }]
    })
  }, "Add assumption"), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Valuation scenarios \xB7 EPS \xD7 P/E"), /*#__PURE__*/React.createElement("p", null, "A simple sensitivity calculation using your inputs. It excludes dividends, FX, dilution changes and timing/discounting. It is not linked to a spreadsheet or consensus feed. Use the same currency, EPS definition and forecast period across scenarios."), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: Object.keys(body.scenarios).length > 0,
    onChange: e => edit({
      ...body,
      scenarios: e.target.checked ? {
        referencePrice: '',
        period: '',
        currency: '',
        asOf: new Date().toISOString().slice(0, 10),
        bearEPS: '',
        bearPE: '',
        baseEPS: '',
        basePE: '',
        bullEPS: '',
        bullPE: ''
      } : {}
    })
  }), "Include valuation scenarios"), Object.keys(body.scenarios).length > 0 && /*#__PURE__*/React.createElement(React.Fragment, null, [['referencePrice', 'Reference share price'], ['period', 'Forecast period and EPS basis'], ['currency', 'Currency'], ['asOf', 'Price as of'], ['bearEPS', 'Bear EPS'], ['bearPE', 'Bear P/E'], ['baseEPS', 'Base EPS'], ['basePE', 'Base P/E'], ['bullEPS', 'Bull EPS'], ['bullPE', 'Bull P/E']].map(([key, label]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("input", {
    type: key === 'asOf' ? 'date' : 'text',
    inputMode: /EPS|PE|Price/.test(key) ? 'decimal' : undefined,
    value: body.scenarios[key] || '',
    onChange: e => edit({
      ...body,
      scenarios: {
        ...body.scenarios,
        [key]: e.target.value
      }
    })
  }))))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    onClick: () => save()
  }, busy ? 'Saving…' : 'Save investment case revision')), !dirty && Object.keys(bridge).length > 0 && /*#__PURE__*/React.createElement("div", {
    style: {
      overflowX: 'auto'
    }
  }, /*#__PURE__*/React.createElement("h3", null, "Saved scenario sensitivities"), /*#__PURE__*/React.createElement("table", null, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Scenario"), /*#__PURE__*/React.createElement("th", null, "Implied price (", body.scenarios.currency, ")"), /*#__PURE__*/React.createElement("th", null, "Price return"))), /*#__PURE__*/React.createElement("tbody", null, Object.entries(bridge).map(([name, v]) => /*#__PURE__*/React.createElement("tr", {
    key: name
  }, /*#__PURE__*/React.createElement("th", null, name), /*#__PURE__*/React.createElement("td", null, v.impliedPrice), /*#__PURE__*/React.createElement("td", null, v.priceReturnPct, "%"))))))), /*#__PURE__*/React.createElement("div", {
    hidden: workspaceTab !== 'reviews'
  }, /*#__PURE__*/React.createElement(ResearchWorkbench, {
    initialKind: "underweight",
    key: active + '-work',
    api: api,
    ticker: active,
    caseRevision: revision,
    assumptions: body.assumptions,
    disabled: busy || dirty
  }), /*#__PURE__*/React.createElement(ResearchDecisions, {
    key: active + '-decisions',
    api: api,
    ticker: active
  })), /*#__PURE__*/React.createElement("div", {
    hidden: workspaceTab !== 'evidence'
  }, /*#__PURE__*/React.createElement(CaseEvidence, {
    key: active,
    revision: revision,
    api: api,
    ticker: active,
    body: body,
    disabled: busy || dirty,
    onCommit: save
  })), workspaceTab === 'signals' && /*#__PURE__*/React.createElement(CaseSignals, {
    key: active + '-signals',
    api: api,
    body: body,
    disabled: busy,
    onChange: edit,
    onSave: save,
    message: message
  }), workspaceTab === 'evolution' && /*#__PURE__*/React.createElement(ThesisEvolution, {
    key: active,
    api: api,
    ticker: active,
    revision: revision
  }), /*#__PURE__*/React.createElement("button", {
    disabled: busy || dirty || !revision,
    onClick: () => {
      var blob = new Blob([JSON.stringify({
        ticker: active,
        revision,
        body
      }, null, 2)], {
        type: 'application/json'
      });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = `${active}-investment-case-r${revision}.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  }, "Download saved case with evidence \xB7 JSON"), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Revision history \xB7 latest ", versions.length), /*#__PURE__*/React.createElement("label", null, "Read a saved revision", /*#__PURE__*/React.createElement("select", {
    value: selectedVersion,
    onChange: e => setSelectedVersion(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose revision"), versions.map(v => /*#__PURE__*/React.createElement("option", {
    key: v.revision,
    value: v.revision
  }, "Revision ", v.revision, " \xB7 ", new Date(v.created_at).toLocaleString())))), snapshot && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", null, "Restore makes a new revision; all intervening history is retained. Review this snapshot before restoring."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || dirty || snapshot.revision === revision,
    onClick: () => save({
      mode: 'restore',
      sourceRevision: snapshot.revision
    })
  }, "Restore revision ", snapshot.revision, " as a new revision"), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Scenario inputs, case signals and evidence metadata"), /*#__PURE__*/React.createElement("pre", {
    style: {
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere'
    }
  }, JSON.stringify({
    scenarios: snapshot.body.scenarios,
    signals: snapshot.body.signals || {},
    evidenceLinks: snapshot.body.evidenceLinks || []
  }, null, 2))), fields.map(([key, label]) => /*#__PURE__*/React.createElement("section", {
    key: key
  }, /*#__PURE__*/React.createElement("h4", null, label), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, snapshot.body[key] || 'Not recorded'))), snapshot.body.assumptions.map(a => /*#__PURE__*/React.createElement("section", {
    key: a.id
  }, /*#__PURE__*/React.createElement("h4", null, a.claim), /*#__PURE__*/React.createElement("p", null, a.evidenceType.replaceAll('_', ' ')), ['support', 'contrary', 'nextTest', 'sourceReference'].map(k => /*#__PURE__*/React.createElement("p", {
    key: k,
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, k, ": ", a[k] || 'Not recorded'))))))));
}