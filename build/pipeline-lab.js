import * as React from 'react';
import { ResearchAssignment } from './research-assignment';
import { labDocuments, visibleLabDocuments } from './pipeline-lab-model.mjs';
var stamp = value => value ? new Date(value).toLocaleDateString('en-US', {
  timeZone: 'America/New_York'
}) : 'Unknown';
async function read(api, path, options = {}) {
  var r = await fetch(api + path, {
    ...options,
    signal: AbortSignal.timeout(25000)
  });
  var d = await r.json();
  if (!r.ok) throw Error(d.error || 'Pipeline Lab could not load.');
  return d;
}
export function PipelineLab({
  api,
  onNavigate
}) {
  var [stocks, setStocks] = React.useState([]),
    [error, setError] = React.useState(''),
    [loaded, setLoaded] = React.useState(false),
    [query, setQuery] = React.useState(''),
    [ticker, setTicker] = React.useState(''),
    [refresh, setRefresh] = React.useState(0);
  React.useEffect(() => {
    var live = true;
    read(api, '/api/pipeline/universe?sort=alpha').then(d => {
      if (!Array.isArray(d.universe)) throw Error('Stock inventory unavailable.');
      if (live) {
        setStocks(d.universe);
        setLoaded(true);
        setError('');
      }
    }).catch(e => {
      if (live) setError(e.message);
    });
    return () => {
      live = false;
    };
  }, [api, refresh]);
  return /*#__PURE__*/React.createElement("div", {
    className: "workspace-page research-desk"
  }, /*#__PURE__*/React.createElement("header", {
    className: "workspace-page-heading"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "EXPERIMENTAL WORKSPACE"), /*#__PURE__*/React.createElement("h1", null, "Research Pipeline Lab"), /*#__PURE__*/React.createElement("p", null, "Your stock-card workflow, with independent source settings. The original Research Pipeline remains available."), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('pipeline')
  }, "Open original Pipeline \u2192")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setRefresh(n => n + 1)
  }, "Retry stock list")), /*#__PURE__*/React.createElement("label", null, "Find a stock ", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Find a Lab stock",
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Ticker or company"
  })), !loaded && !error && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading stocks\u2026"), /*#__PURE__*/React.createElement("div", {
    className: "workspace-create-grid",
    style: {
      marginTop: 20
    }
  }, stocks.filter(s => `${s.ticker} ${s.company}`.toLowerCase().includes(query.toLowerCase())).map(s => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: s.ticker
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, s.sector || 'Coverage'), /*#__PURE__*/React.createElement("h2", null, s.ticker), /*#__PURE__*/React.createElement("p", null, s.company), /*#__PURE__*/React.createElement("p", null, "Thesis updated: ", stamp(s.lastAnalysisDate)), /*#__PURE__*/React.createElement("p", null, s.docCount || 0, " stored documents \xB7 External-source freshness not yet checked here"), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    onClick: () => setTicker(s.ticker)
  }, "Sources & settings")))), loaded && !stocks.length && /*#__PURE__*/React.createElement("p", null, "No stocks are in the Pipeline universe yet. Add one in the original Pipeline."), ticker && /*#__PURE__*/React.createElement("div", {
    className: "fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-2 sm:p-6"
  }, /*#__PURE__*/React.createElement("section", {
    role: "dialog",
    "aria-modal": "true",
    "aria-label": `${ticker} Lab sources`,
    className: "workspace-panel",
    style: {
      width: '100%',
      maxWidth: 1100,
      maxHeight: '92vh',
      overflowY: 'auto'
    }
  }, /*#__PURE__*/React.createElement(LabStock, {
    key: ticker,
    api: api,
    ticker: ticker,
    onNavigate: onNavigate,
    onClose: () => setTicker('')
  }))));
}
function LabStock({
  api,
  ticker,
  onNavigate,
  onClose
}) {
  var [rows, setRows] = React.useState([]),
    [saved, setSaved] = React.useState(null),
    [selection, setSelection] = React.useState({
      thesis: [],
      note: []
    }),
    [task, setTask] = React.useState('thesis'),
    [query, setQuery] = React.useState(''),
    [group, setGroup] = React.useState('All'),
    [sort, setSort] = React.useState('date'),
    [error, setError] = React.useState(''),
    [notice, setNotice] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [generation, setGeneration] = React.useState(0),
    [collect, setCollect] = React.useState(false),
    [checkResult, setCheckResult] = React.useState(null),
    [checking, setChecking] = React.useState(false);
  var lock = React.useRef(false),
    alive = React.useRef(true);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  React.useEffect(() => {
    var live = true;
    setSaved(null);
    setError('');
    Promise.all([read(api, `/api/pipeline/documents/${ticker}`), read(api, `/api/agent/local-files/${ticker}`), read(api, `/api/pipeline-lab/config/${ticker}`)]).then(([a, b, c]) => {
      var list = labDocuments(a.documents, b.files);
      if (!Number.isInteger(c.revision) || !Array.isArray(c.selection?.thesis) || !Array.isArray(c.selection?.note)) throw Error('Saved Lab settings unavailable.');
      if (live) {
        setRows(list);
        setSaved(c);
        setSelection(c.selection);
      }
    }).catch(e => {
      if (live) setError(e.message);
    });
    return () => {
      live = false;
    };
  }, [api, ticker, generation]);
  var dirty = saved && JSON.stringify(selection) !== JSON.stringify(saved.selection);
  var close = () => {
    if (!dirty || window.confirm('Leave without saving your Lab selection changes?')) onClose();
  };
  var save = async () => {
    if (lock.current || !saved) return;
    lock.current = true;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      var d = await read(api, `/api/pipeline-lab/config/${ticker}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          revision: saved.revision,
          selection
        })
      });
      if (alive.current) {
        setSaved(d);
        setSelection(d.selection);
        setNotice('Lab settings saved centrally for all devices. No research was started.');
      }
    } catch (e) {
      if (alive.current) setError(e.message + ' Your selections remain on this screen.');
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var check = async () => {
    if (lock.current || !saved || dirty) return;
    lock.current = true;
    setChecking(true);
    setCheckResult(null);
    setError('');
    try {
      var d = await read(api, `/api/pipeline-lab/check/${ticker}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          revision: saved.revision,
          task
        })
      });
      if (alive.current) setCheckResult(d);
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setChecking(false);
    }
  };
  var shown = visibleLabDocuments(rows, query, group, sort),
    selected = new Set(selection[task]);
  var missing = selection[task].filter(k => !rows.some(r => r.key === k));
  var toggle = key => setSelection(s => ({
    ...s,
    [task]: s[task].includes(key) ? s[task].filter(k => k !== key) : [...s[task], key]
  }));
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "INDEPENDENT LAB CONFIGURATION"), /*#__PURE__*/React.createElement("h2", null, ticker, " \xB7 Sources & settings")), /*#__PURE__*/React.createElement("button", {
    "aria-label": "Close Lab settings",
    onClick: close,
    disabled: busy
  }, "Close")), /*#__PURE__*/React.createElement("p", null, "Organize sources here without moving iCloud files. Thesis and note selections are saved separately. This first Lab release saves research selections; they are not yet connected to generation."), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, /*#__PURE__*/React.createElement("button", {
    "aria-pressed": task === 'thesis',
    onClick: () => setTask('thesis')
  }, "Thesis sources (", selection.thesis.length, ")"), /*#__PURE__*/React.createElement("button", {
    "aria-pressed": task === 'note',
    onClick: () => setTask('note')
  }, "Note sources (", selection.note.length, ")")), /*#__PURE__*/React.createElement("p", null, task === 'thesis' ? 'Choose evidence that changes or tests the investment case, including relevant foundational work.' : 'Choose evidence for the specific event or question your research note should address.'), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error), notice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => {
      if (!dirty || window.confirm('Reload and discard unsaved Lab selections?')) {
        setNotice('');
        setGeneration(n => n + 1);
      }
    }
  }, "Reload saved settings & inventory"), !saved ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, error ? 'Inventory is incomplete; saving is disabled.' : 'Loading sources and saved configuration…') : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions",
    style: {
      marginTop: 16,
      flexWrap: 'wrap'
    }
  }, /*#__PURE__*/React.createElement("input", {
    "aria-label": "Search Lab sources",
    placeholder: "Search title or folder",
    value: query,
    onChange: e => setQuery(e.target.value)
  }), /*#__PURE__*/React.createElement("select", {
    "aria-label": "Source type",
    value: group,
    onChange: e => setGroup(e.target.value)
  }, ['All', 'Transcripts', 'Filings', 'Releases', 'Presentations', 'Other / unclassified'].map(x => /*#__PURE__*/React.createElement("option", {
    key: x
  }, x))), /*#__PURE__*/React.createElement("select", {
    "aria-label": "Source sort",
    value: sort,
    onChange: e => setSort(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "date"
  }, "Newest available date"), /*#__PURE__*/React.createElement("option", {
    value: "name"
  }, "Filename A\u2013Z"))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Types are suggested from filenames. \u201CAdded\u201D and \u201CFile modified\u201D are not publication dates. Charlie and iCloud copies remain separate until their original bytes can be matched.")), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setSelection(s => ({
      ...s,
      [task]: [...new Set([...s[task], ...shown.map(r => r.key)])]
    }))
  }, "Select visible (", shown.length, ")"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setSelection(s => ({
      ...s,
      [task]: []
    }))
  }, "Clear ", task, " selection"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setSelection(s => ({
      ...s,
      [task]: [...s[task === 'thesis' ? 'note' : 'thesis']]
    }))
  }, "Copy ", task === 'thesis' ? 'note' : 'thesis', " selection")), missing.length > 0 && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, missing.length, " saved selection(s) are absent from the current inventory. They remain saved; review before running research."), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 16
    }
  }, shown.map(r => /*#__PURE__*/React.createElement("label", {
    key: r.key,
    style: {
      display: 'flex',
      gap: 12,
      padding: '14px 0',
      borderBottom: '1px solid var(--border-color, #8884)',
      alignItems: 'flex-start'
    }
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    disabled: busy,
    checked: selected.has(r.key),
    onChange: () => toggle(r.key),
    "aria-label": `Select ${r.filename} from ${r.location} ${r.folder}`
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      minWidth: 0,
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("strong", null, r.filename), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, r.type, " \xB7 ", r.dateLabel, ": ", stamp(r.date), " \xB7 ", r.location, " / ", r.folder), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, r.ready ? 'Stored in Charlie — exact-byte verification not shown by this inventory' : 'Listed in iCloud — import readiness not established', r.repeated ? ' · Same filename exists elsewhere; kept separate' : ''))))), !shown.length && /*#__PURE__*/React.createElement("p", null, "No sources match this view."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    style: {
      marginTop: 16
    },
    disabled: busy || !dirty,
    onClick: save
  }, busy ? 'Saving…' : 'Save Lab settings'), /*#__PURE__*/React.createElement("button", {
    style: {
      marginTop: 16,
      marginLeft: 8
    },
    disabled: busy || checking || dirty || !saved,
    onClick: check
  }, checking ? 'Checking stored originals…' : 'Check selected originals'), dirty && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Save your selection before checking the stored originals.")), checkResult && !dirty && checkResult.task === task && checkResult.revision === saved.revision && /*#__PURE__*/React.createElement("div", {
    role: "status"
  }, /*#__PURE__*/React.createElement("p", null, checkResult.verified.length, " selected original(s) checked against stored bytes."), checkResult.blockers.length > 0 ? /*#__PURE__*/React.createElement("ul", null, checkResult.blockers.map((b, i) => /*#__PURE__*/React.createElement("li", {
    key: i,
    style: {
      overflowWrap: 'anywhere'
    }
  }, b))) : /*#__PURE__*/React.createElement("p", null, "The selected stored bytes passed this check."), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "This does not verify external-source freshness, licensing or equality with iCloud originals. No research was started. Generation from Lab selections is still being built; originals must be rechecked when a run starts.")))), /*#__PURE__*/React.createElement("hr", {
    style: {
      margin: '24px 0'
    }
  }), /*#__PURE__*/React.createElement("h3", null, "Collect latest sources & prepare outputs"), /*#__PURE__*/React.createElement("p", null, "This uses Charlie\u2019s existing coordinated assignment workflow and its own source policy, not the saved Lab selections above. It can collect up to eight eligible originals and prepare reviewable outputs. Browser collection still depends on the Mac and signed-in AlphaSense."), /*#__PURE__*/React.createElement("button", {
    onClick: () => setCollect(v => !v)
  }, collect ? 'Hide assignment controls' : 'Open coordinated assignment controls'), collect && /*#__PURE__*/React.createElement(ResearchAssignment, {
    api: api,
    ticker: ticker,
    onReport: () => onNavigate('stockanalysis', ticker)
  }), /*#__PURE__*/React.createElement("p", null, "For the current direct-update and Generate Note workflow, ", /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      if (!dirty || window.confirm('Leave unsaved Lab changes?')) onNavigate('pipeline', ticker);
    }
  }, "open original Research Pipeline \u2192")));
}