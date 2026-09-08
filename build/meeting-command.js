import { AssignmentWorkspace } from './assignment-workspace';
import * as React from 'react';
var today = () => new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/New_York'
}).format(new Date());
var focusOptions = [['thesis', 'Investment thesis'], ['earnings', 'Earnings & guidance'], ['competition', 'Competition & demand'], ['capital', 'Capital allocation'], ['followups', 'Previous meeting follow-ups']];
var pendingKey = 'charlie-meeting-request-v1';
function stored() {
  try {
    return JSON.parse(sessionStorage.getItem(pendingKey) || 'null');
  } catch {
    return null;
  }
}
export function MeetingCommand({
  api,
  onNavigate,
  onSection,
  sourcePolicy
}) {
  var [tickers, setTickers] = React.useState([]),
    [selected, setSelected] = React.useState([]),
    [query, setQuery] = React.useState('');
  var [meetingDate, setMeetingDate] = React.useState(today),
    [days, setDays] = React.useState(90),
    [focuses, setFocuses] = React.useState(['thesis', 'earnings', 'followups']),
    [note, setNote] = React.useState('');
  var [reuseJob, setReuseJob] = React.useState('');
  var [format, setFormat] = React.useState('conference'),
    [audience, setAudience] = React.useState('specialist');
  var formatLabels = {
    conference: '30-minute conference · 12–15 questions',
    one_on_one: '60-minute 1×1 · 25–30 questions',
    hosted_pm: 'Hosted PM discussion · 35–45 questions'
  };
  var [jobs, setJobs] = React.useState([]),
    [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState(''),
    [message, setMessage] = React.useState(''),
    [uncertain, setUncertain] = React.useState(() => !!stored());
  var formRef = React.useRef(null);
  var pending = React.useRef(stored()),
    lock = React.useRef(false),
    alive = React.useRef(true);
  var json = async (path, options = {}) => {
    var r = await fetch(api + path, {
      ...options,
      signal: AbortSignal.timeout(20000)
    });
    var d;
    try {
      d = await r.json();
    } catch {
      throw Error('The server response could not be confirmed.');
    }
    if (!r.ok) {
      var e = Error(d.error || `Request failed (${r.status})`);
      e.status = r.status;
      throw e;
    }
    return d;
  };
  var clear = () => {
    pending.current = null;
    try {
      sessionStorage.removeItem(pendingKey);
    } catch {}
    setUncertain(false);
  };
  var refresh = async () => {
    var d = await json('/api/research/meeting-commands');
    if (!alive.current) return;
    setTickers(d.tickers);
    setJobs(d.jobs);
    setError('');
    if (pending.current && d.jobs.some(j => j.batch_id === pending.current.requestId || j.id === pending.current.requestId)) {
      clear();
      setMessage('Your meeting assignment is recorded. Follow each company below.');
    }
  };
  React.useEffect(() => {
    alive.current = true;
    refresh().catch(e => setError(e.message));
    var timer = setInterval(() => {
      if (!document.hidden) refresh().catch(e => {
        if (alive.current) setError(e.message);
      });
    }, 10000);
    return () => {
      alive.current = false;
      clearInterval(timer);
    };
  }, [api]);
  var toggle = (v, list, set) => set(list.includes(v) ? list.filter(x => x !== v) : [...list, v]);
  var submit = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    var body = pending.current || {
      requestId: crypto.randomUUID(),
      tickers: selected,
      meetingDate,
      date: today(),
      days,
      focuses,
      note,
      format,
      audience,
      sourcePolicy,
      reuseJob
    };
    pending.current = body;
    try {
      sessionStorage.setItem(pendingKey, JSON.stringify(body));
    } catch {}
    try {
      var d = await json(body.reuseJob ? `/api/research/meeting-commands/${body.reuseJob}/reuse` : '/api/research/meeting-commands', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });
      clear();
      setMessage(`Queued ${d.commands?.length || 1} meeting pack${!d.commands || d.commands.length === 1 ? '' : 's'}. You can leave this page; progress is saved.`);
      await refresh();
    } catch (e) {
      setError(e.message);
      if (e.status && e.status < 500) clear();else setUncertain(true);
    } finally {
      lock.current = false;
      setBusy(false);
    }
  };
  var retry = async id => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    try {
      await json(`/api/research/meeting-commands/${id}/retry`, {
        method: 'POST'
      });
      await refresh();
      setMessage('Meeting pack queued to resume from its saved checkpoint.');
    } catch (e) {
      setError(e.message);
    } finally {
      lock.current = false;
      setBusy(false);
    }
  };
  var issueFor = j => j.prep_error || j.error || (j.meeting_issue === 'The latest linked analyst activity has no completed recap yet' ? '' : j.meeting_issue);
  return /*#__PURE__*/React.createElement("section", {
    ref: formRef,
    className: "workspace-panel meeting-command"
  }, /*#__PURE__*/React.createElement("div", {
    className: "meeting-command-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "MEETING & CONFERENCE PREP"), /*#__PURE__*/React.createElement("h2", null, "Walk in with better questions."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Choose your companies. Charlie collects recent research, prepares a brief and saves questions in Meeting Prep. No prompt required.")), /*#__PURE__*/React.createElement("span", {
    className: "meeting-command-count"
  }, selected.length, /*#__PURE__*/React.createElement("small", null, "of 10 companies"))), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || uncertain,
    className: "meeting-command-form"
  }, /*#__PURE__*/React.createElement("div", {
    className: "meeting-command-step"
  }, /*#__PURE__*/React.createElement("h3", null, /*#__PURE__*/React.createElement("span", null, "1"), " Choose companies"), /*#__PURE__*/React.createElement("label", null, "Source collection", /*#__PURE__*/React.createElement("select", {
    value: reuseJob,
    onChange: e => {
      setReuseJob(e.target.value);
      var j = jobs.find(j => j.job_id === e.target.value);
      if (j) setSelected([j.ticker]);
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Find fresh sources in AlphaSense"), jobs.filter(j => j.prep_status === 'done').map(j => /*#__PURE__*/React.createElement("option", {
    key: j.job_id,
    value: j.job_id
  }, "Reuse ", j.ticker, " pack \xB7 ", j.options?.meetingPrep?.meetingDate, " \xB7 sources through ", j.options?.until)))), reuseJob && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Uses this pack\u2019s verified originals and original source preferences. No new search or Mac connection needed. Creates a separate meeting; original pack stays saved. Cached analyses are reused when eligible; generation uses model credits."), /*#__PURE__*/React.createElement("label", null, "Find a covered company", /*#__PURE__*/React.createElement("input", {
    type: "search",
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Search ticker, e.g. ABT"
  })), /*#__PURE__*/React.createElement("div", {
    className: "meeting-ticker-grid"
  }, tickers.filter(t => t.toLowerCase().includes(query.toLowerCase())).map(t => /*#__PURE__*/React.createElement("label", {
    key: t,
    className: selected.includes(t) ? 'selected' : ''
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: selected.includes(t),
    disabled: !!reuseJob || selected.length >= 10 && !selected.includes(t),
    onChange: () => toggle(t, selected, setSelected)
  }), t))), !tickers.length && /*#__PURE__*/React.createElement("p", null, "Coverage loads from your analyst team. Assign a covering analyst before requesting a pack."), selected.length > 0 && /*#__PURE__*/React.createElement("p", {
    className: "meeting-selection"
  }, "Selected: ", /*#__PURE__*/React.createElement("strong", null, selected.join(' · ')), " ", /*#__PURE__*/React.createElement("button", {
    type: "button",
    disabled: !!reuseJob,
    onClick: () => setSelected([])
  }, "Clear selection"))), /*#__PURE__*/React.createElement("div", {
    className: "meeting-command-step"
  }, /*#__PURE__*/React.createElement("h3", null, /*#__PURE__*/React.createElement("span", null, "2"), " Set the brief"), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Meeting date", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: meetingDate,
    onChange: e => setMeetingDate(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Source lookback", /*#__PURE__*/React.createElement("select", {
    disabled: !!reuseJob,
    value: days,
    onChange: e => setDays(Number(e.target.value))
  }, /*#__PURE__*/React.createElement("option", {
    value: 30
  }, "Past 30 days"), /*#__PURE__*/React.createElement("option", {
    value: 60
  }, "Past 60 days"), /*#__PURE__*/React.createElement("option", {
    value: 90
  }, "Past 90 days \xB7 default")))), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Meeting format", /*#__PURE__*/React.createElement("select", {
    value: format,
    onChange: e => setFormat(e.target.value)
  }, Object.entries(formatLabels).map(([id, label]) => /*#__PURE__*/React.createElement("option", {
    key: id,
    value: id
  }, label)))), /*#__PURE__*/React.createElement("label", null, "Audience", /*#__PURE__*/React.createElement("select", {
    value: audience,
    onChange: e => setAudience(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "specialist"
  }, "Sector specialists"), /*#__PURE__*/React.createElement("option", {
    value: "generalist"
  }, "Generalist portfolio managers")))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, format === 'conference' ? 'Focused on the most important investment debates and recent developments.' : 'Broader coverage of strategy, business economics, competition, growth, margins, capital allocation and risks, alongside recent events.', " Every pack marks must-ask questions; the rest are optional follow-ups. Counts are targets, subject to available evidence."), /*#__PURE__*/React.createElement("p", null, "What matters most? ", /*#__PURE__*/React.createElement("small", null, "Optional\u2014start with the selected defaults.")), /*#__PURE__*/React.createElement("div", {
    className: "meeting-focuses"
  }, focusOptions.map(([id, label]) => /*#__PURE__*/React.createElement("label", {
    key: id
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: focuses.includes(id),
    onChange: () => toggle(id, focuses, setFocuses)
  }), label))), /*#__PURE__*/React.createElement("label", null, "Anything specific? ", /*#__PURE__*/React.createElement("small", null, "Optional"), /*#__PURE__*/React.createElement("textarea", {
    maxLength: 1000,
    value: note,
    onChange: e => setNote(e.target.value),
    placeholder: "For example: focus on FreeStyle Libre adoption and margin durability."
  }))), /*#__PURE__*/React.createElement("div", {
    className: "meeting-command-preview"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "YOUR ASSIGNMENT"), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, formatLabels[format]), " \xB7 ", audience === 'generalist' ? 'Generalist PM audience' : 'Specialist audience'), /*#__PURE__*/React.createElement("p", null, "Prepare ", selected.length ? selected.join(', ') : 'your selected companies', " for ", meetingDate || 'your meeting date', ". ", reuseJob ? `Reuse verified originals from ${jobs.find(j => j.job_id === reuseJob)?.options?.since || 'the saved window'} through ${jobs.find(j => j.job_id === reuseJob)?.options?.until || 'the saved cutoff'}.` : `Seek earnings transcripts, presentations and sell-side reaction from the past ${days} days.`, " Prepare prioritized questions with source references and follow-ups."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, reuseJob ? 'Verified saved originals → background meeting preparation. No fresh collection; source dates and coverage remain those of the selected pack.' : 'Collection → verified originals → analyst brief → meeting questions. Requires your Mac and signed-in AlphaSense browser; collection pickup is scheduled.', " Uses model credits; missing evidence is disclosed."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    type: "button",
    disabled: !sourcePolicy || !selected.length || !meetingDate,
    onClick: submit
  }, busy ? 'Queuing…' : selected.length > 1 ? `Prepare ${selected.length} company packs →` : 'Prepare meeting pack →'))), uncertain && /*#__PURE__*/React.createElement("div", {
    role: "status"
  }, /*#__PURE__*/React.createElement("p", null, "Submission is not yet confirmed. Retry checks the same request; it does not create a new batch."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: submit
  }, "Check / retry same assignment")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h3", null, "Your meeting packs"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('meetingprep')
  }, "Open Meeting Prep \u2192")), !jobs.length && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "Your assignments will appear here, with separate progress for every company."), /*#__PURE__*/React.createElement("div", {
    className: "meeting-pack-list"
  }, jobs.map(j => /*#__PURE__*/React.createElement("article", {
    key: j.id
  }, /*#__PURE__*/React.createElement(AssignmentWorkspace, {
    api: api,
    id: j.id,
    onNavigate: onNavigate,
    onSection: onSection
  }), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, j.ticker), /*#__PURE__*/React.createElement("span", null, j.options?.meetingPrep?.meetingDate)), /*#__PURE__*/React.createElement("p", null, j.prep_status === 'done' ? 'Pack ready' : j.prep_status === 'running' ? `Preparing questions · ${j.prep_step || 'starting'}${j.completed ? ` (${j.completed}/${j.total})` : ''}` : j.prep_status === 'queued' ? 'Sources verified · waiting for preparation' : j.prep_status === 'failed' ? 'Preparation needs attention' : j.status === 'failed' ? 'Collection request failed' : j.status === 'queued' ? 'Waiting for Mac acknowledgment' : 'Collecting sources / preparing analyst brief'), issueFor(j) && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error"
  }, issueFor(j)), j.meeting_id && /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      window.dispatchEvent(new CustomEvent('charlie-open-meeting', {
        detail: {
          id: Number(j.meeting_id)
        }
      }));
      onNavigate('meetingprep');
    }
  }, "Open ", j.ticker, " meeting \u2192"), j.prep_status === 'failed' && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => retry(j.job_id)
  }, "Retry meeting pack"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: () => {
      setSelected([j.ticker]);
      setReuseJob(j.prep_status === 'done' ? j.job_id : '');
      setFormat('one_on_one');
      setAudience(j.options?.meetingPrep?.audience || 'specialist');
      setMessage('Choose the format and audience above, then submit a new brief. Your existing pack stays saved.');
      formRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  }, "Prepare a different format \u2192"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onSection('collection')
  }, "Collection details \u2192")))));
}