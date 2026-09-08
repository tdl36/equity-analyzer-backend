import * as React from 'react';
var kinds = [['filing', '8-K and reaction'], ['earnings', 'Earnings review'], ['event', 'Event investigation']];
var today = () => new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/New_York'
}).format(new Date());
export function CommandCharlie({
  api,
  onNavigate,
  onSection
}) {
  var [ticker, setTicker] = React.useState(''),
    [date, setDate] = React.useState(today),
    [kind, setKind] = React.useState('filing'),
    [days, setDays] = React.useState(1),
    [coordinated, setCoordinated] = React.useState(true),
    [instruction, setInstruction] = React.useState('');
  var [fav, setFav] = React.useState({
      favorites: [],
      revision: ''
    }),
    [favId, setFavId] = React.useState(''),
    [name, setName] = React.useState(''),
    [jobs, setJobs] = React.useState([]);
  var [error, setError] = React.useState(''),
    [favoriteError, setFavoriteError] = React.useState(''),
    [macDate, setMacDate] = React.useState(null),
    [message, setMessage] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [uncertain, setUncertain] = React.useState(false);
  var alive = React.useRef(true),
    lock = React.useRef(false),
    pending = React.useRef(null);
  var json = async (path, options = {}) => {
    var r = await fetch(`${api}${path}`, {
      ...options,
      signal: AbortSignal.timeout(20000)
    });
    var d = await r.json();
    if (!r.ok) {
      var e = Error(d.error || `Request failed (${r.status})`);
      e.status = r.status;
      throw e;
    }
    return d;
  };
  var refresh = async () => {
    try {
      var d = await json('/api/research/commands');
      if (alive.current) {
        setJobs(d.jobs);
        setMacDate(d.macReportedAt);
        setError('');
        if (pending.current && d.jobs.some(j => j.id === pending.current.requestId)) {
          pending.current = null;
          setUncertain(false);
          setMessage('Task recorded. Follow its progress below.');
        }
      }
    } catch (e) {
      if (alive.current) setError(e.message);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    refresh();
    json('/api/research/command-favorites').then(d => {
      if (alive.current) setFav(d);
    }).catch(e => {
      if (alive.current) setFavoriteError(e.message);
    });
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 10000);
    return () => {
      alive.current = false;
      clearInterval(timer);
    };
  }, [api]);
  var pick = f => {
    setFavId(f.id);
    setName(f.name);
    setInstruction(f.instruction);
    setKind(f.kind);
    setDays(f.days);
    setCoordinated(f.coordinated === true);
  };
  var run = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setMessage('Submitting task…');
    var body = pending.current || {
      requestId: crypto.randomUUID(),
      ticker: ticker.trim().toUpperCase(),
      date,
      kind,
      days,
      instruction,
      coordinated
    };
    pending.current = body;
    try {
      var d = await json('/api/research/commands', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });
      pending.current = null;
      setUncertain(false);
      setMessage(`Task queued for ${d.plan.ticker}: ${d.plan.since} through ${d.plan.until}. You can close this page and return to its saved status.`);
      await refresh();
    } catch (e) {
      setMessage(`Submission not confirmed: ${e.message}`);
      if (e.status) pending.current = null;else setUncertain(true);
      await refresh();
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var save = async remove => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    try {
      var row = {
        id: favId || crypto.randomUUID(),
        name,
        instruction,
        kind,
        days,
        coordinated
      };
      var values = remove ? fav.favorites.filter(f => f.id !== favId) : [...fav.favorites.filter(f => f.id !== row.id), row];
      var d = await json('/api/research/command-favorites', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          revision: fav.revision,
          favorites: values
        })
      });
      setFav(d);
      setFavId(remove ? '' : row.id);
      setMessage(remove ? 'Favorite removed.' : 'Favorite saved across your devices.');
    } catch (e) {
      setMessage(e.message);
      if (e.status === 409) setFav(await json('/api/research/command-favorites'));
    } finally {
      lock.current = false;
      setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel command-charlie"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "COMMAND CHARLIE"), /*#__PURE__*/React.createElement("h2", null, "Give the team a research assignment."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Start with a repeatable workflow or write your own instructions. The ticker, event date and collection window below define the task. Collection starts when the Mac browser worker picks it up; research drafts remain subject to review."), /*#__PURE__*/React.createElement("div", {
    className: "command-favorites"
  }, fav.favorites.map(f => /*#__PURE__*/React.createElement("button", {
    key: f.id,
    "aria-pressed": favId === f.id,
    disabled: busy || uncertain,
    onClick: () => pick(f)
  }, /*#__PURE__*/React.createElement("strong", null, f.name), /*#__PURE__*/React.createElement("small", null, kinds.find(k => k[0] === f.kind)?.[1], " \xB7 ", f.days, " day", f.days === 1 ? '' : 's')))), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || uncertain,
    className: "desk-form"
  }, /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Ticker", /*#__PURE__*/React.createElement("input", {
    value: ticker,
    maxLength: 20,
    onChange: e => setTicker(e.target.value.toUpperCase()),
    placeholder: "UNH"
  })), /*#__PURE__*/React.createElement("label", null, "Event date (New York)", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: date,
    max: today(),
    onChange: e => setDate(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Lookback days", /*#__PURE__*/React.createElement("input", {
    type: "number",
    min: "1",
    max: "90",
    value: days,
    onChange: e => setDays(Number(e.target.value))
  })), /*#__PURE__*/React.createElement("label", null, "Research workflow", /*#__PURE__*/React.createElement("select", {
    value: kind,
    onChange: e => setKind(e.target.value)
  }, kinds.map(([k, l]) => /*#__PURE__*/React.createElement("option", {
    key: k,
    value: k
  }, l))))), /*#__PURE__*/React.createElement("label", null, "Your assignment", /*#__PURE__*/React.createElement("textarea", {
    rows: 5,
    maxLength: 3000,
    value: instruction,
    onChange: e => setInstruction(e.target.value),
    placeholder: "Review UNH\u2019s 8-K today and sell-side reaction. Explain what changed and propose updates to my thesis."
  })), instruction && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Assignment preview:"), " ", instruction.replace(/\{ticker\}/g, ticker.trim() || '[choose ticker]').replace(/\{date\}/g, date || '[choose date]')), /*#__PURE__*/React.createElement("p", null, "SEC EDGAR filings + AlphaSense press releases, broker research and transcripts \u2192 CATALYSTS event folder \u2192 covering analyst recap. Missing sources and retrieval failures are reported explicitly. Thesis changes are proposed for your review, not automatically applied."), /*#__PURE__*/React.createElement("label", {
    className: "command-team-option"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: coordinated,
    onChange: e => setCoordinated(e.target.checked)
  }), /*#__PURE__*/React.createElement("span", null, "Include independent challenge and editorial revision ", /*#__PURE__*/React.createElement("small", null, "Up to two additional model passes. Final selected-claim source review follows."))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: !ticker.trim() || !instruction.trim(),
    onClick: run
  }, "Run research task \u2197"), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Save or edit a repeatable favorite"), /*#__PURE__*/React.createElement("p", null, "Use ", '{ticker}', " and ", '{date}', " in the instruction to reuse it for another company or event date."), /*#__PURE__*/React.createElement("label", null, "Favorite name", /*#__PURE__*/React.createElement("input", {
    value: name,
    maxLength: 80,
    onChange: e => setName(e.target.value)
  })), /*#__PURE__*/React.createElement("button", {
    disabled: !name.trim() || !instruction.trim(),
    onClick: () => save(false)
  }, "Save favorite"), /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      setFavId('');
      setName('');
    }
  }, "Start a new favorite"), favId && /*#__PURE__*/React.createElement("button", {
    disabled: fav.favorites.length <= 1,
    onClick: () => save(true)
  }, "Remove selected favorite"))), uncertain && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: run
  }, "Retry same task request"), favoriteError && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, "Favorites: ", favoriteError), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h3", null, "Task history"), /*#__PURE__*/React.createElement("button", {
    onClick: refresh
  }, "Refresh tasks")), /*#__PURE__*/React.createElement("p", null, "Mac last reported: ", macDate ? new Date(macDate).toLocaleString() : 'not yet available', ". Tasks wait for an available, signed-in browser worker."), !jobs.length && /*#__PURE__*/React.createElement("p", null, error ? 'Task history is unavailable. Retry when the connection returns.' : 'No research assignments have been submitted yet.'), jobs.map(j => /*#__PURE__*/React.createElement("article", {
    className: "amendment-card",
    key: j.id
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h3", null, j.ticker, " \xB7 ", j.input.payload.kind, " research"), /*#__PURE__*/React.createElement("span", {
    className: "desk-status"
  }, j.collection?.status || j.status)), /*#__PURE__*/React.createElement("p", null, j.input.payload.instruction), /*#__PURE__*/React.createElement("p", null, j.input.payload.since, " through ", j.input.payload.until), /*#__PURE__*/React.createElement("p", null, "Mac command: ", j.status, j.collection ? ` · Browser collection: ${j.collection.status}` : j.status === 'applied' ? ' · Browser status absent from the latest Mac snapshot' : ' · Awaiting Mac acknowledgement'), j.collection?.result?.research && typeof j.collection.result.research === 'string' && /*#__PURE__*/React.createElement("p", null, j.collection.result.research), j.error && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error"
  }, j.error), j.collection?.issue && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error"
  }, j.collection.issue), /*#__PURE__*/React.createElement("p", null, j.reports?.length ? j.reports.map(r => `${r.has_report ? 'Recap available' : r.status} (${r.id.slice(0, 8)})${r.recovery_attempts ? ` · Recovery ${r.recovery_attempts}/2` : ''}${!r.has_report && r.current_step ? ` · ${r.current_step}` : ''}`).join(' · ') : 'No linked analyst recap is available yet.'), j.reports?.some(r => r.roles?.length) && /*#__PURE__*/React.createElement("p", null, "Team stages: ", j.reports.flatMap(r => r.roles || []).map(r => `${r.role}: ${r.status.replaceAll('_', ' ')}`).join(' · ')), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Execution plan and limits"), /*#__PURE__*/React.createElement("ol", null, j.input.payload.steps.map(s => /*#__PURE__*/React.createElement("li", {
    key: s
  }, s))), j.input.payload.limitations.map(s => /*#__PURE__*/React.createElement("p", {
    key: s
  }, s))), /*#__PURE__*/React.createElement("button", {
    onClick: () => onSection('collection')
  }, "Collection and recovery controls \u2192"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('analysts')
  }, "Open Analyst inbox \u2192"))));
}