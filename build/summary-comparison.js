import React from 'react';
var labels = {
  brief: 'Executive brief',
  takeaways: 'Key takeaways',
  record: 'Management record',
  questions: 'Follow-up questions',
  assessment: 'Investment assessment'
};
var originalKeys = {
  brief: 'brief',
  takeaways: 'summary',
  record: 'meeting_summary',
  questions: 'questions',
  assessment: 'assessment'
};
export function SummaryComparison({
  summary,
  api,
  getKey,
  renderHtml
}) {
  var [open, setOpen] = React.useState(false),
    [rows, setRows] = React.useState([]),
    [selected, setSelected] = React.useState(''),
    [section, setSection] = React.useState('takeaways'),
    [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState(''),
    [feedback, setFeedback] = React.useState(''),
    [saved, setSaved] = React.useState(false);
  var epoch = React.useRef(0);
  React.useEffect(() => {
    epoch.current++;
    setRows([]);
    setSelected('');
    setOpen(false);
    setError('');
  }, [summary.id]);
  var base = `${api}/api/summaries/${encodeURIComponent(summary.id)}/comparisons`;
  React.useEffect(() => {
    if (!open) return;
    var alive = true;
    var timer;
    async function poll() {
      try {
        var r = await fetch(base, {
          signal: AbortSignal.timeout(20000)
        });
        var d = await r.json();
        if (!r.ok) throw Error(d.error || 'Comparison could not be loaded.');
        if (alive) {
          setRows(d.comparisons);
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
  }, [base, open]);
  var row = rows.find(r => r.id === selected) || rows[0];
  React.useEffect(() => {
    setFeedback(row?.feedback || '');
    setSaved(false);
  }, [row?.id]);
  var state = row?.state || {};
  async function start(resumeId) {
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
          resumeId
        }),
        signal: AbortSignal.timeout(25000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Could not start comparison.');
      if (token !== epoch.current) return;
      setSelected(d.id);
      var fresh = await fetch(base);
      if (fresh.ok && token === epoch.current) setRows((await fresh.json()).comparisons);
    } catch (e) {
      if (token === epoch.current) setError(e.message);
    } finally {
      if (token === epoch.current) setBusy(false);
    }
  }
  var record = Object.keys(state.parts || {}).sort((a, b) => Number(a) - Number(b)).map(k => state.parts[k].record).join('\n\n');
  var text = section === 'record' ? record : state.sections?.[section];
  return /*#__PURE__*/React.createElement("section", {
    className: "rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 my-4"
  }, /*#__PURE__*/React.createElement("button", {
    className: "text-left w-full flex justify-between gap-3",
    "aria-expanded": open,
    onClick: () => setOpen(!open)
  }, /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("strong", null, "Compare improved notes"), /*#__PURE__*/React.createElement("small", {
    className: "block text-slate-400 mt-1"
  }, "Trial workspace \xB7 Your original note stays intact")), /*#__PURE__*/React.createElement("span", null, open ? '−' : '+')), open && /*#__PURE__*/React.createElement("div", {
    className: "mt-4 space-y-4"
  }, /*#__PURE__*/React.createElement("p", {
    className: "text-sm text-slate-400"
  }, "Generate an alternative from the complete saved source. Long transcripts are read in parts, with every part retained in the management record. This starts paid AI generation; it does not transcribe the audio again."), /*#__PURE__*/React.createElement("button", {
    className: "px-3 py-2 rounded-lg bg-amber-600 text-white disabled:opacity-50",
    disabled: busy || !summary.rawNotes?.trim(),
    onClick: () => start()
  }, busy ? 'Starting…' : rows.length ? 'Open or generate comparison for current source' : 'Generate improved comparison'), !summary.rawNotes?.trim() && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "No saved source text is available for this note yet."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "text-red-400"
  }, error), row && /*#__PURE__*/React.createElement(React.Fragment, null, rows.length > 1 && /*#__PURE__*/React.createElement("label", {
    className: "block"
  }, "Comparison version ", /*#__PURE__*/React.createElement("select", {
    className: "bg-slate-900 p-2 rounded",
    value: row.id,
    onChange: e => setSelected(e.target.value)
  }, rows.map(r => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, r.version, " \xB7 ", new Date(r.created_at).toLocaleString())))), /*#__PURE__*/React.createElement("div", {
    role: "status",
    className: "text-sm"
  }, /*#__PURE__*/React.createElement("strong", null, row.status === 'complete' ? 'Ready for comparison' : row.status === 'failed' ? 'Needs retry' : state.progress || 'Queued'), /*#__PURE__*/React.createElement("p", {
    className: "text-slate-400"
  }, (state.coveredCharacters || 0).toLocaleString(), " / ", (state.sourceCharacters || summary.rawNotes?.length || 0).toLocaleString(), " source characters processed \xB7 ", Object.keys(state.parts || {}).length, " / ", state.totalParts || '—', " parts saved \xB7 Last update ", new Date(row.updated_at).toLocaleTimeString())), row.error && /*#__PURE__*/React.createElement("p", {
    className: "text-red-400"
  }, row.error), row.status !== 'complete' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    className: "underline",
    onClick: () => start(row.id)
  }, "Resume saved comparison"), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400"
  }, "Safe to resume after interruption; an active worker cannot be started twice. Saved sections are reused.")), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400"
  }, "Left: original output frozen when this comparison began. Right: ", row.version, ". No prior thesis/model comparison or independent factual verification. ", state.hierarchicalSynthesis ? 'Long-source synthesis uses consolidated evidence; inspect the full part records for detail.' : ''), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2",
    "aria-label": "Comparison section"
  }, Object.entries(labels).map(([key, label]) => /*#__PURE__*/React.createElement("button", {
    key: key,
    "aria-pressed": section === key,
    className: `px-3 py-2 rounded-lg text-sm ${section === key ? 'bg-amber-600 text-white' : 'bg-white/5'}`,
    onClick: () => setSection(key)
  }, label))), /*#__PURE__*/React.createElement("div", {
    className: "grid grid-cols-1 xl:grid-cols-2 gap-4"
  }, /*#__PURE__*/React.createElement("article", {
    className: "min-w-0 rounded-lg border border-white/10 p-4"
  }, /*#__PURE__*/React.createElement("h4", {
    className: "font-semibold mb-3"
  }, "Original \xB7 ", labels[section]), /*#__PURE__*/React.createElement("div", {
    className: "prose prose-invert max-w-none text-sm break-words",
    dangerouslySetInnerHTML: {
      __html: renderHtml(row.baseline?.[originalKeys[section]] || '<p>No original section saved.</p>')
    }
  })), /*#__PURE__*/React.createElement("article", {
    className: "min-w-0 rounded-lg border border-amber-500/20 p-4"
  }, /*#__PURE__*/React.createElement("h4", {
    className: "font-semibold mb-3"
  }, "Improved \xB7 ", labels[section]), /*#__PURE__*/React.createElement("div", {
    className: "whitespace-pre-wrap break-words text-sm leading-relaxed"
  }, text || 'This section has not been generated yet. Saved management-record parts are available as processing progresses.'))), /*#__PURE__*/React.createElement("label", {
    className: "block text-sm font-medium"
  }, "Comparison notes", /*#__PURE__*/React.createElement("textarea", {
    className: "block w-full bg-transparent border border-white/20 rounded-lg p-3 mt-2",
    rows: 3,
    maxLength: 10000,
    value: feedback,
    onChange: e => {
      setFeedback(e.target.value);
      setSaved(false);
    },
    placeholder: "What is better? What detail was lost? Which version would you use?"
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
          })
        });
        if (!r.ok) throw Error('Comparison notes could not be saved.');
        setSaved(true);
      } catch (e) {
        setError(e.message);
      }
    }
  }, "Save comparison notes"), saved && /*#__PURE__*/React.createElement("span", {
    role: "status",
    className: "text-sm ml-3"
  }, "Saved"))));
}