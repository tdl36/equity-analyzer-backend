import React from 'react';
var sections = [['brief', 'Executive brief', 'brief'], ['takeaways', 'Key takeaways', 'summary'], ['record', 'Management record', 'meeting_summary'], ['questions', 'Follow-up questions', 'questions'], ['assessment', 'Investment assessment', 'assessment']];
export function SummaryComparison({
  summary,
  api,
  getKey,
  renderHtml,
  children
}) {
  var [view, setView] = React.useState('original');
  var [rows, setRows] = React.useState([]),
    [loaded, setLoaded] = React.useState(false),
    [selected, setSelected] = React.useState('');
  var [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState(''),
    [feedback, setFeedback] = React.useState(''),
    [saved, setSaved] = React.useState(false);
  var epoch = React.useRef(0);
  var base = `${api}/api/summaries/${encodeURIComponent(summary.id)}/comparisons`;
  React.useEffect(() => {
    epoch.current++;
    setRows([]);
    setLoaded(false);
    setSelected('');
    setView('original');
    setError('');
  }, [summary.id]);
  React.useEffect(() => {
    var alive = true,
      timer;
    async function poll() {
      try {
        var r = await fetch(base, {
          signal: AbortSignal.timeout(20000)
        });
        var d = await r.json();
        if (!r.ok) throw Error(d.error || 'Improved notes could not be loaded.');
        if (alive) {
          setRows(d.comparisons);
          setLoaded(true);
          setError('');
        }
      } catch (e) {
        if (alive) setError(e.message);
      } finally {
        if (alive) timer = setTimeout(poll, 8000);
      }
    }
    poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [base]);
  var row = rows.find(r => r.id === selected) || rows[0],
    state = row?.state || {};
  React.useEffect(() => {
    setFeedback(row?.feedback || '');
    setSaved(false);
  }, [row?.id]);
  var running = row && ['queued', 'running'].includes(row.status);
  var stale = running && Date.now() - new Date(row.updated_at).getTime() > 10 * 60 * 1000;
  async function start() {
    var token = epoch.current;
    setBusy(true);
    setError('');
    try {
      var r = await fetch(base, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          apiKey: getKey(),
          resumeId: row?.id
        }),
        signal: AbortSignal.timeout(25000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Could not start improved notes.');
      if (token !== epoch.current) return;
      setSelected(d.id);
      var fresh = await fetch(base, {
        signal: AbortSignal.timeout(20000)
      });
      if (!fresh.ok) throw Error('Could not refresh note status.');
      var data = await fresh.json();
      if (token === epoch.current) setRows(data.comparisons);
    } catch (e) {
      if (token === epoch.current) setError(e.message);
    } finally {
      if (token === epoch.current) setBusy(false);
    }
  }
  var record = Object.keys(state.parts || {}).sort((a, b) => Number(a) - Number(b)).map(k => state.parts[k].record).join('\n\n');
  var status = row?.status === 'complete' ? 'Ready' : row?.status === 'failed' ? 'Needs retry' : running ? 'Generating' : loaded ? 'Not generated' : 'Loading';
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("section", {
    className: "rounded-xl border border-white/15 p-4 my-4",
    "aria-label": "Note versions"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap items-center justify-between gap-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2",
    role: "group",
    "aria-label": "Choose note version"
  }, [['original', 'Original'], ['improved', 'Improved'], ['compare', 'Side by side']].map(([key, label]) => /*#__PURE__*/React.createElement("button", {
    key: key,
    "aria-pressed": view === key,
    onClick: () => setView(key),
    className: `px-4 py-2 rounded-lg font-medium text-sm ${view === key ? 'bg-amber-600 text-white' : 'bg-white/5 hover:bg-white/10'}`
  }, label))), /*#__PURE__*/React.createElement("span", {
    role: "status",
    className: "text-sm text-slate-400"
  }, "Improved notes \xB7 ", status)), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400 mt-3"
  }, "New summaries automatically generate both versions. Your original notes stay intact.")), view === 'original' ? children : /*#__PURE__*/React.createElement("section", {
    className: "space-y-5",
    "aria-label": view === 'compare' ? 'Side-by-side notes' : 'Improved notes'
  }, error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "text-red-400"
  }, error), !loaded && !error && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading improved notes\u2026"), loaded && !row && /*#__PURE__*/React.createElement("div", {
    className: "rounded-xl border border-white/15 p-5"
  }, /*#__PURE__*/React.createElement("h3", {
    className: "font-semibold"
  }, "Generate the improved version for this saved note"), /*#__PURE__*/React.createElement("p", {
    className: "text-sm text-slate-400 my-3"
  }, "Older notes need one initial run. Charlie uses the saved transcript; no audio upload is needed. This uses your research API credits."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !summary.rawNotes?.trim(),
    onClick: start,
    className: "bg-amber-600 text-white rounded-lg px-4 py-2 disabled:opacity-50"
  }, busy ? 'Starting…' : 'Generate improved notes'), !summary.rawNotes?.trim() && /*#__PURE__*/React.createElement("p", {
    className: "mt-2 text-sm"
  }, "No saved source text is available.")), row && /*#__PURE__*/React.createElement(React.Fragment, null, rows.length > 1 && /*#__PURE__*/React.createElement("label", {
    className: "block text-sm"
  }, "Saved version ", /*#__PURE__*/React.createElement("select", {
    className: "bg-transparent border border-white/20 p-2 rounded",
    value: row.id,
    onChange: e => setSelected(e.target.value)
  }, rows.map(r => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, new Date(r.created_at).toLocaleString(), " \xB7 ", r.status)))), running && /*#__PURE__*/React.createElement("div", {
    role: "status",
    className: "rounded-xl border border-amber-500/30 p-4"
  }, /*#__PURE__*/React.createElement("strong", null, state.progress || 'Queued for generation'), /*#__PURE__*/React.createElement("p", {
    className: "text-sm text-slate-400 mt-1"
  }, "You can leave this page. Completed sections appear below as they are saved.")), (row.status === 'failed' || stale) && /*#__PURE__*/React.createElement("div", {
    role: "alert",
    className: "rounded-xl border border-amber-500/30 p-4"
  }, /*#__PURE__*/React.createElement("p", null, row.error || 'No recent progress. Resume from the last saved checkpoint.'), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: start,
    className: "underline mt-2"
  }, busy ? 'Starting…' : 'Retry improved notes')), view === 'compare' && /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400"
  }, "Original on the left, improved on the right. The original is the saved snapshot from when this version began. On mobile, each pair is stacked."), sections.map(([key, label, original]) => {
    var text = key === 'record' ? record : state.sections?.[key];
    return /*#__PURE__*/React.createElement("section", {
      key: key,
      className: "space-y-3"
    }, /*#__PURE__*/React.createElement("h3", {
      className: "text-lg font-semibold"
    }, label), /*#__PURE__*/React.createElement("div", {
      className: `grid gap-4 ${view === 'compare' ? 'xl:grid-cols-2' : 'grid-cols-1'}`
    }, view === 'compare' && /*#__PURE__*/React.createElement("article", {
      className: "min-w-0 rounded-xl border border-white/10 p-5"
    }, /*#__PURE__*/React.createElement("h4", {
      className: "text-xs uppercase tracking-wide text-slate-400 mb-4"
    }, "Original"), /*#__PURE__*/React.createElement("div", {
      className: "prose prose-invert max-w-none text-sm break-words",
      dangerouslySetInnerHTML: {
        __html: renderHtml(row.baseline?.[original] || '<p>No original section saved.</p>')
      }
    })), /*#__PURE__*/React.createElement("article", {
      className: "min-w-0 rounded-xl border border-amber-500/20 p-5"
    }, view === 'compare' && /*#__PURE__*/React.createElement("h4", {
      className: "text-xs uppercase tracking-wide text-amber-500 mb-4"
    }, "Improved"), /*#__PURE__*/React.createElement("div", {
      className: "whitespace-pre-wrap break-words text-sm leading-relaxed"
    }, text || 'Waiting for this section…'))));
  }), /*#__PURE__*/React.createElement("details", {
    className: "text-sm text-slate-400"
  }, /*#__PURE__*/React.createElement("summary", {
    className: "cursor-pointer"
  }, "Source coverage and method"), /*#__PURE__*/React.createElement("p", {
    className: "mt-2"
  }, (state.coveredCharacters || 0).toLocaleString(), " / ", (state.sourceCharacters || summary.rawNotes?.length || 0).toLocaleString(), " source characters processed. Full management records are retained. Interpretations are AI analysis, not independent verification or your own views. ", state.hierarchicalSynthesis ? 'Long-source synthesis uses consolidated evidence.' : '')), /*#__PURE__*/React.createElement("label", {
    className: "block text-sm font-medium"
  }, "Your comparison feedback", /*#__PURE__*/React.createElement("textarea", {
    className: "block w-full bg-transparent border border-white/20 rounded-lg p-3 mt-2",
    rows: 3,
    maxLength: 10000,
    value: feedback,
    onChange: e => {
      setFeedback(e.target.value);
      setSaved(false);
    },
    placeholder: "Which version works better? What should Charlie preserve or improve?"
  })), /*#__PURE__*/React.createElement("button", {
    className: "underline text-sm",
    onClick: async () => {
      try {
        var r = await fetch(`${base}/${row.id}/feedback`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            feedback
          }),
          signal: AbortSignal.timeout(20000)
        });
        if (!r.ok) throw Error('Feedback could not be saved.');
        setSaved(true);
      } catch (e) {
        setError(e.message);
      }
    }
  }, "Save feedback"), saved && /*#__PURE__*/React.createElement("span", {
    role: "status",
    className: "text-sm ml-3"
  }, "Saved"))));
}