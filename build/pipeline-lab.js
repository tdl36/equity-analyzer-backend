import { checkedResearchImport, checkedLabImportRequest } from './lab-import-receipts.mjs';
import LabNote from './pipeline-lab-note';
import LabICloudImport from './pipeline-lab-import';
import * as React from 'react';
import { parseTimestamp, workId } from './workspace-model.mjs';
import { ResearchAssignment } from './research-assignment';
import { labDocuments, visibleLabDocuments, recentLabDocuments, labCollectionStatus, taskSourceGuide, groupLabDocuments, sameLabSelection, labSourceDate, labTicker } from './pipeline-lab-model.mjs';
var stamp = value => parseTimestamp(value)?.toLocaleDateString('en-US', {
  timeZone: 'America/New_York'
}) || 'Unknown';
async function read(api, path, options = {}) {
  var r = await fetch(api + path, {
    ...options,
    signal: AbortSignal.timeout(25000)
  });
  var d = await r.json();
  if (!r.ok) {
    var e = Error(d.error || 'Pipeline Lab could not load.');
    e.status = r.status;
    throw e;
  }
  return d;
}
export function PipelineLab({
  api,
  onNavigate,
  renderHtml,
  initialTicker = '',
  routeWork = ''
}) {
  var [openSymbol, setOpenSymbol] = React.useState(''),
    [symbolError, setSymbolError] = React.useState('');
  var [stocks, setStocks] = React.useState([]),
    [error, setError] = React.useState(''),
    [loaded, setLoaded] = React.useState(false),
    [query, setQuery] = React.useState(''),
    [ticker, setTicker] = React.useState(initialTicker),
    [refresh, setRefresh] = React.useState(0),
    [collection, setCollection] = React.useState(null),
    [collectionError, setCollectionError] = React.useState('');
  React.useEffect(() => {
    setTicker(initialTicker);
  }, [initialTicker, routeWork]);
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
  React.useEffect(() => {
    var live = true;
    var load = () => read(api, '/api/collection/control').then(d => {
      if (live) {
        setCollection(d);
        setCollectionError('');
      }
    }).catch(() => {
      if (live) setCollectionError('Collection status unavailable. Open collection recovery before assuming sources are current.');
    });
    load();
    var timer = setInterval(load, 60000);
    return () => {
      live = false;
      clearInterval(timer);
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
  }, "Retry stock list")), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, collectionError || 'Collection status comes from the Mac’s last report. It is not a live AlphaSense feed.', " ", collection?.updatedAt ? 'Reported ' + new Date(collection.updatedAt).toLocaleString('en-US', {
    timeZone: 'America/New_York'
  }) : '')), /*#__PURE__*/React.createElement("label", null, "Find a stock ", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Find a Lab stock",
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Ticker or company"
  })), /*#__PURE__*/React.createElement("form", {
    "aria-label": "Open a stock in Lab",
    onSubmit: e => {
      e.preventDefault();
      try {
        var symbol = labTicker(openSymbol);
        setSymbolError('');
        onNavigate('pipelinelab', symbol);
      } catch (e) {
        setSymbolError(e.message);
      }
    },
    style: {
      marginTop: 16
    }
  }, /*#__PURE__*/React.createElement("label", null, "Open a ticker directly", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Ticker to open in Lab",
    value: openSymbol,
    maxLength: 20,
    autoCapitalize: "characters",
    autoCorrect: "off",
    spellCheck: false,
    placeholder: "e.g. ABT",
    onChange: e => {
      setOpenSymbol(e.target.value);
      setSymbolError('');
    }
  })), /*#__PURE__*/React.createElement("button", {
    type: "submit",
    disabled: !openSymbol.trim()
  }, "Open ticker in Lab"), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Works even when a stock is not in these cards. Opening shows its source settings and collection controls; it does not add a thesis, search AlphaSense or start paid research. Confirm the company and collection choices before starting.")), symbolError && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, symbolError)), !loaded && !error && /*#__PURE__*/React.createElement("p", {
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
  }, s.sector || 'Coverage'), /*#__PURE__*/React.createElement("h2", null, s.ticker), /*#__PURE__*/React.createElement("p", null, s.company), /*#__PURE__*/React.createElement("p", null, "Thesis updated: ", stamp(s.lastAnalysisDate)), /*#__PURE__*/React.createElement("p", null, s.docCount || 0, " stored documents"), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, labCollectionStatus(collection, s.ticker))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    onClick: () => setTicker(s.ticker)
  }, "Sources & settings")))), loaded && !stocks.length && /*#__PURE__*/React.createElement("p", null, "No stocks are in the Pipeline universe yet. Open a ticker directly above to inspect its Lab workspace."), ticker && /*#__PURE__*/React.createElement("div", {
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
    focusId: ticker === initialTicker ? workId(routeWork, 'labnote') : '',
    renderHtml: renderHtml,
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
  onClose,
  renderHtml,
  focusId
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
    [recency, setRecency] = React.useState('all'),
    [error, setError] = React.useState(''),
    [notice, setNotice] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [generation, setGeneration] = React.useState(0),
    [collect, setCollect] = React.useState(false),
    [checkResult, setCheckResult] = React.useState(null),
    [checking, setChecking] = React.useState(false),
    [pendingImport, setPendingImport] = React.useState(false),
    [selectedOnly, setSelectedOnly] = React.useState(false),
    [publishedOnly, setPublishedOnly] = React.useState(false),
    [grouped, setGrouped] = React.useState(true),
    [latest, setLatest] = React.useState(null);
  var [noteUnsaved, setNoteUnsaved] = React.useState(false);
  var lock = React.useRef(false),
    alive = React.useRef(true),
    sourceHeading = React.useRef(null);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  React.useEffect(() => {
    var live = true;
    setSaved(null);
    setLatest(null);
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
  var importKey = 'charlie-lab-import-v1:' + api + ':' + ticker;
  React.useEffect(() => {
    try {
      setPendingImport(!!sessionStorage.getItem(importKey));
    } catch {
      setError('Import retry storage is unavailable. Imports cannot start safely.');
    }
  }, [importKey]);
  var dirty = saved && !sameLabSelection(selection, saved.selection);
  var collected = explicit => {
    setCheckResult(null);
    if (dirty) {
      setNotice('Collection completed. Your unsaved selections are preserved. Compare with latest saved selections or reload when ready to review newly imported originals.');
    } else {
      setNotice('Collection completed. Source inventory is being refreshed; review dates, coverage and task relevance before selecting originals.');
      setGeneration(n => n + 1);
    }
    if (explicit) sourceHeading.current?.scrollIntoView({
      block: 'start'
    });
  };
  var downloadOriginal = async row => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var r = await fetch(api + `/api/pipeline-lab/original/${encodeURIComponent(ticker)}/${encodeURIComponent(row.id)}`, {
        signal: AbortSignal.timeout(30000)
      });
      if (!r.ok) {
        var d = await r.json();
        throw Error(d.error || 'Original unavailable.');
      }
      var url = URL.createObjectURL(await r.blob());
      var a = document.createElement('a');
      a.href = url;
      a.download = row.filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var compare = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var d = await read(api, `/api/pipeline-lab/config/${ticker}`);
      if (!Number.isInteger(d.revision) || !Array.isArray(d.selection?.thesis) || !Array.isArray(d.selection?.note)) throw Error('Latest settings unavailable. Your edits remain on this screen.');
      if (alive.current) setLatest(d);
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var adopt = keep => {
    if (!latest) return;
    setSaved(latest);
    if (!keep) setSelection(latest.selection);
    setLatest(null);
    setCheckResult(null);
    setNotice(keep ? 'Your selections are retained against the latest saved version. Review them and Save Lab settings to explicitly replace the saved selections. No save or research was started.' : 'Latest saved selections loaded. No research was started.');
  };
  var close = () => {
    if (!dirty && !noteUnsaved || window.confirm('Leave without saving your Lab source selections or note edits?')) onClose();
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
        setLatest(null);
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
  var importResearch = async () => {
    if (lock.current || !saved || dirty) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var old = sessionStorage.getItem(importKey);
      var body = old ? JSON.parse(old) : {
        requestId: crypto.randomUUID(),
        revision: saved.revision,
        task
      };
      checkedLabImportRequest(body);
      if (!old) sessionStorage.setItem(importKey, JSON.stringify(body));
      if (sessionStorage.getItem(importKey) !== JSON.stringify(body)) throw Error('Could not preserve import identity. No import was sent.');
      setPendingImport(true);
      var d = await read(api, `/api/pipeline-lab/import-research/${ticker}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });
      checkedResearchImport(d, ticker, body, saved.revision === body.revision ? saved.selection[body.task] : null);
      sessionStorage.removeItem(importKey);
      if (alive.current) {
        setPendingImport(false);
        setCheckResult(null);
        setNotice(`${d.receipts.length} Research-storage original(s) matched exact bytes in Charlie. This task’s selection was updated; no research was started.`);
        setGeneration(n => n + 1);
      }
    } catch (e) {
      if ([400, 409].includes(e.status)) {
        sessionStorage.removeItem(importKey);
        if (alive.current) setPendingImport(false);
      }
      if (alive.current) setError(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var shown = visibleLabDocuments(recentLabDocuments(rows, recency, Date.now(), publishedOnly), query, group, sort).filter(r => !selectedOnly || selection[task].includes(r.key)),
    selected = new Set(selection[task]);
  var missing = selection[task].filter(k => !rows.some(r => r.key === k));
  var toggle = key => setSelection(s => ({
    ...s,
    [task]: s[task].includes(key) ? s[task].filter(k => k !== key) : [...s[task], key]
  }));
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    ref: sourceHeading,
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "INDEPENDENT LAB CONFIGURATION"), /*#__PURE__*/React.createElement("h2", null, ticker, " \xB7 Sources & settings")), /*#__PURE__*/React.createElement("button", {
    "aria-label": "Close Lab settings",
    onClick: close,
    disabled: busy
  }, "Close")), /*#__PURE__*/React.createElement("p", null, "Organize sources here without moving iCloud files. Thesis and note selections are saved separately. Check your saved selection to prepare separate research drafts. Browser collection remains available below."), /*#__PURE__*/React.createElement("div", {
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
  }, "Reload saved settings & inventory"), dirty && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: compare
  }, "Compare with latest saved selections"), latest && /*#__PURE__*/React.createElement("section", {
    "aria-label": "Compare saved selections",
    className: "workspace-panel",
    style: {
      marginTop: 16
    }
  }, /*#__PURE__*/React.createElement("h3", null, "Review saved changes before saving"), /*#__PURE__*/React.createElement("p", null, "Saved version ", latest.revision, " compared with your current choices. Imported originals may have different storage identities. Keeping your choices can leave newly imported originals unselected."), ['thesis', 'note'].map(t => /*#__PURE__*/React.createElement("div", {
    key: t
  }, /*#__PURE__*/React.createElement("h4", null, t === 'thesis' ? 'Thesis' : 'Note', " sources"), /*#__PURE__*/React.createElement("p", null, selection[t].length, " on this screen \xB7 ", latest.selection[t].length, " saved"), [['Only on this screen', selection[t].filter(k => !latest.selection[t].includes(k))], ['Only in saved settings', latest.selection[t].filter(k => !selection[t].includes(k))]].map(([label, keys]) => /*#__PURE__*/React.createElement("details", {
    key: label
  }, /*#__PURE__*/React.createElement("summary", null, label, " (", keys.length, ")"), /*#__PURE__*/React.createElement("ul", null, keys.map(k => /*#__PURE__*/React.createElement("li", {
    key: k,
    style: {
      overflowWrap: 'anywhere'
    }
  }, rows.find(r => r.key === k)?.filename || k, /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, k)))))))), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => adopt(true)
  }, "Keep my choices for review"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => {
      if (window.confirm('Replace your unsaved selections with the latest saved selections?')) adopt(false);
    }
  }, "Use latest saved selections"), /*#__PURE__*/React.createElement("button", {
    onClick: () => setLatest(null)
  }, "Close comparison")), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Keeping your choices does not save them. The next save still checks for concurrent changes. Neither action imports sources or starts research."))), !saved ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, error ? 'Inventory is incomplete; saving is disabled.' : 'Loading sources and saved configuration…') : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions",
    style: {
      marginTop: 16,
      flexWrap: 'wrap'
    }
  }, /*#__PURE__*/React.createElement("input", {
    "aria-label": "Search Lab sources",
    placeholder: "Search title, publisher or folder",
    value: query,
    onChange: e => setQuery(e.target.value)
  }), /*#__PURE__*/React.createElement("select", {
    "aria-label": "Source type",
    value: group,
    onChange: e => setGroup(e.target.value)
  }, ['All', 'Transcripts', 'Filings', 'Releases', 'Presentations', 'Broker research', 'Other / unclassified'].map(x => /*#__PURE__*/React.createElement("option", {
    key: x
  }, x))), /*#__PURE__*/React.createElement("select", {
    "aria-label": "Source recency",
    value: recency,
    onChange: e => setRecency(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "all"
  }, "All available dates"), /*#__PURE__*/React.createElement("option", {
    value: "30"
  }, "Last 30 days"), /*#__PURE__*/React.createElement("option", {
    value: "90"
  }, "Last 90 days"), /*#__PURE__*/React.createElement("option", {
    value: "365"
  }, "Last year")), /*#__PURE__*/React.createElement("select", {
    "aria-label": "Source sort",
    value: sort,
    onChange: e => setSort(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "date"
  }, "Newest available date"), /*#__PURE__*/React.createElement("option", {
    value: "name"
  }, "Filename A\u2013Z"))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Types use recorded collection metadata when available; otherwise they are suggested from filenames. Recency uses the labeled date; undated sources remain in \u201CAll available dates\u201D. \u201CAdded\u201D and \u201CFile modified\u201D are not publication dates. Charlie and iCloud copies remain separate until their original bytes can be matched.")), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: grouped,
    onChange: e => setGrouped(e.target.checked)
  }), " Group sources by document type")), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: publishedOnly,
    onChange: e => setPublishedOnly(e.target.checked)
  }), " Use recorded publication dates only"), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, "Excludes added/modified dates, unknown dates and future dates from this view. Combine with Last 30 days to review recently published evidence. Hidden selections remain selected.")), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: selectedOnly,
    onChange: e => setSelectedOnly(e.target.checked)
  }), " Review selected sources only (", selection[task].length, ")")), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, selection[task].filter(k => !shown.some(r => r.key === k)).length, " selected source(s) outside this view, including missing sources. Filters do not change the saved selection.")), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setSelection(s => ({
      ...s,
      [task]: [...new Set([...s[task], ...shown.map(r => r.key)])]
    }))
  }, "Select visible (", shown.length, ")"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !shown.some(r => selected.has(r.key)),
    onClick: () => setSelection(s => ({
      ...s,
      [task]: s[task].filter(k => !shown.some(r => r.key === k))
    }))
  }, "Deselect visible"), /*#__PURE__*/React.createElement("button", {
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
  }, groupLabDocuments(shown, grouped).map(g => /*#__PURE__*/React.createElement("section", {
    key: g.label,
    "aria-label": g.label + ' sources',
    style: {
      marginBottom: 24
    }
  }, /*#__PURE__*/React.createElement("h3", null, g.label, " \xB7 ", g.rows.length), g.rows.map(r => /*#__PURE__*/React.createElement("div", {
    key: r.key,
    style: {
      display: 'flex',
      gap: 12,
      padding: '14px 0',
      borderBottom: '1px solid var(--border-color, #8884)',
      alignItems: 'flex-start'
    }
  }, /*#__PURE__*/React.createElement("input", {
    id: `lab-source-${r.key}`,
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
  }, /*#__PURE__*/React.createElement("label", {
    htmlFor: `lab-source-${r.key}`
  }, /*#__PURE__*/React.createElement("strong", null, r.filename)), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, r.type, r.typeRecorded ? ' (recorded)' : ' (suggested)', " \xB7 ", r.dateLabel, ": ", labSourceDate(r.date), " \xB7 ", r.location, " / ", r.folder, r.publisher ? ' · ' + r.publisher : '', r.author ? ' · ' + r.author : ''), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, r.ready ? 'Stored in Charlie — exact-byte verification not shown by this inventory' : 'Listed in iCloud — import readiness not established', r.repeated ? ' · Same filename exists elsewhere; kept separate' : ''), r.location === 'Charlie' && String(r.id).startsWith('thesis_') && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => downloadOriginal(r)
  }, "Download original for review")), /*#__PURE__*/React.createElement("details", {
    style: {
      marginTop: 8
    }
  }, /*#__PURE__*/React.createElement("summary", null, "How to use for ", task === 'note' ? 'a note' : 'the thesis'), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, taskSourceGuide(r.type, task))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Guidance based on the displayed document type, not a review of its contents. Recency alone does not establish relevance; selection remains yours."))), r.originalReceiptRecorded && /*#__PURE__*/React.createElement("details", {
    onClick: e => e.stopPropagation(),
    style: {
      marginTop: 8
    }
  }, /*#__PURE__*/React.createElement("summary", null, "Recorded browser collection receipt"), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "This records an earlier import, not a fresh byte check or proof that the search found every relevant source. Use Check selected originals before generation.")), r.collectionRun && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Collection run: ", r.collectionRun)), r.recordedSha256 && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Recorded SHA-256: ", r.recordedSha256)), r.sourceUrl && /*#__PURE__*/React.createElement("a", {
    href: r.sourceUrl,
    target: "_blank",
    rel: "noopener noreferrer",
    onClick: e => e.stopPropagation()
  }, "Open observed AlphaSense source \u2197")))))))), !shown.length && /*#__PURE__*/React.createElement("p", null, "No sources match this view."), (pendingImport || selection[task].some(r => r.startsWith('charlie:research_'))) && /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 16
    }
  }, /*#__PURE__*/React.createElement("p", null, "Import selected Research-storage originals into Charlie using exact byte checks. Existing originals stay in place. Different same-named files block the whole import. Use the iCloud import controls below for files listed only on the Mac."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || checking || dirty,
    onClick: importResearch
  }, pendingImport ? 'Retry same Research import' : 'Import selected Research originals')), /*#__PURE__*/React.createElement(LabICloudImport, {
    api: api,
    ticker: ticker,
    task: task,
    revision: saved.revision,
    selected: selection[task],
    disabled: busy || checking || dirty,
    read: read,
    onImported: () => {
      setCheckResult(null);
      if (dirty) {
        setNotice('An iCloud import completed while you were editing. Your unsaved selections are preserved on this screen. Reload saved settings & inventory when ready to review the imported originals; this will replace unsaved selections.');
      } else {
        setGeneration(n => n + 1);
      }
    }
  }), /*#__PURE__*/React.createElement("button", {
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
  }, b))) : /*#__PURE__*/React.createElement("p", null, "The selected stored bytes passed this check."), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "This does not verify external-source freshness, licensing or equality with iCloud originals. No research was started. Originals are rechecked when a run starts. Generated outputs remain drafts for your review."))), checkResult && !dirty && !checkResult.blockers.length && checkResult.task === task && checkResult.revision === saved.revision && checkResult.generationAvailable !== true && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "The connected server does not yet support generation from Lab selections. Refresh after the backend update finishes."), task === 'thesis' && checkResult?.generationAvailable === true && !dirty && !checkResult.blockers.length && checkResult.task === task && checkResult.revision === saved.revision && /*#__PURE__*/React.createElement(ResearchAssignment, {
    embedded: true,
    key: `${task}:${saved.revision}:${checkResult.digest}`,
    api: api,
    ticker: ticker,
    labSelection: {
      task,
      revision: saved.revision,
      digest: checkResult.digest
    },
    onReport: () => {
      if (!noteUnsaved || window.confirm('Leave unsaved note edits?')) onNavigate('stockanalysis', ticker);
    }
  })), /*#__PURE__*/React.createElement(LabNote, {
    onUnsavedChange: setNoteUnsaved,
    key: focusId || ticker,
    focusId: focusId,
    api: api,
    ticker: ticker,
    read: read,
    renderHtml: renderHtml,
    selection: task === 'note' && !dirty && checkResult?.generationAvailable === true && !checkResult.blockers.length && checkResult.task === task && checkResult.revision === saved?.revision ? {
      revision: saved.revision,
      digest: checkResult.digest
    } : null
  }), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    onClick: e => {
      if (noteUnsaved && !window.confirm('Leave unsaved note edits?')) e.preventDefault();
    },
    href: `#view=desk&ticker=${encodeURIComponent(ticker)}&section=collection`
  }, "Open this stock\u2019s browser collection schedule, history & recovery \u2192")), /*#__PURE__*/React.createElement("hr", {
    style: {
      margin: '24px 0'
    }
  }), /*#__PURE__*/React.createElement("h3", null, "Collect latest sources through AlphaSense in Chrome"), /*#__PURE__*/React.createElement("p", null, "No AlphaSense API or MCP is required. The scheduled Operations worker uses your signed-in Chrome session. This uses Charlie\u2019s existing coordinated assignment workflow and its own source policy, not the saved Lab selections above. It can collect up to eight eligible originals and prepare reviewable outputs. Keep the Mac awake and online, its agent running, and Codex and signed-in Chrome available. Sign-in or MFA is completed directly in AlphaSense. Collection is not guaranteed to be immediate."), /*#__PURE__*/React.createElement("button", {
    onClick: () => setCollect(v => !v)
  }, collect ? 'Hide assignment controls' : 'Open coordinated assignment controls'), collect && /*#__PURE__*/React.createElement(ResearchAssignment, {
    onCollectionComplete: collected,
    labMode: true,
    embedded: true,
    api: api,
    ticker: ticker,
    onReport: () => {
      if (!noteUnsaved || window.confirm('Leave unsaved note edits?')) onNavigate('stockanalysis', ticker);
    }
  }), /*#__PURE__*/React.createElement("p", null, "For the current direct-update and Generate Note workflow, ", /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      if (!dirty && !noteUnsaved || window.confirm('Leave unsaved Lab source selections or note edits?')) onNavigate('pipeline', ticker);
    }
  }, "open original Research Pipeline \u2192")));
}