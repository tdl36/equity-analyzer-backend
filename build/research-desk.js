import { ResearchHistory } from './research-history';
import { PortfolioPriorities } from './portfolio-priorities';
import { CollectionCloud } from './collection-cloud';
import { EarningsWorkspace } from './earnings-workspace';
import { EvidenceWorkspace } from './evidence-workspace';
import * as React from 'react';
import { parseTickers, researchQueue, runCounts, runsNeedingStatusCheck, PLAYBOOKS } from './research-desk-model.mjs';
import { parseTimestamp } from './workspace-model.mjs';
var {
  useState,
  useEffect,
  useRef
} = React;
var stamp = v => parseTimestamp(v)?.toLocaleString('en-US', {
  month: 'short',
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit'
}) || 'Date unavailable';
function Status({
  value
}) {
  return /*#__PURE__*/React.createElement("span", {
    className: `desk-status desk-status-${value}`
  }, String(value || 'unknown').replaceAll('_', ' '));
}
function ReportContent({
  value,
  renderHtml,
  depth = 0
}) {
  if (value == null) return /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "No report attached yet.");
  if (typeof value === 'object' && depth < 4) return Array.isArray(value) ? /*#__PURE__*/React.createElement("div", null, value.map((v, i) => /*#__PURE__*/React.createElement(ReportContent, {
    key: i,
    value: v,
    renderHtml: renderHtml,
    depth: depth + 1
  }))) : /*#__PURE__*/React.createElement("div", null, Object.entries(value).filter(([, v]) => v != null).map(([key, v]) => /*#__PURE__*/React.createElement("section", {
    className: "desk-report-section",
    key: key
  }, /*#__PURE__*/React.createElement("h3", null, key.replace(/([a-z])([A-Z])/g, '$1 $2').replaceAll('_', ' ')), /*#__PURE__*/React.createElement(ReportContent, {
    value: v,
    renderHtml: renderHtml,
    depth: depth + 1
  }))));
  var text = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
  return renderHtml ? /*#__PURE__*/React.createElement("div", {
    className: "desk-report-prose",
    dangerouslySetInnerHTML: {
      __html: renderHtml(text)
    }
  }) : /*#__PURE__*/React.createElement("pre", null, text);
}
export function ResearchDesk({
  api,
  analyses,
  onCompany,
  onNavigate,
  renderHtml,
  renderRecapHtml
}) {
  var [data, setData] = useState({
    runs: [],
    activities: [],
    failed: [],
    analysts: [],
    providers: {},
    capabilities: {
      maxConcurrency: 1,
      maxBatchSize: 12
    }
  });
  var [errors, setErrors] = useState([]),
    [loading, setLoading] = useState(true),
    [updated, setUpdated] = useState(null);
  var [section, setSection] = useState('overview'),
    [days, setDays] = useState(90),
    [query, setQuery] = useState('');
  var [tickers, setTickers] = useState(''),
    [provider, setProvider] = useState('anthropic'),
    [model, setModel] = useState('');
  var [date, setDate] = useState(new Date().toLocaleDateString('en-CA')),
    [concurrency, setConcurrency] = useState(1);
  var [review, setReview] = useState(false),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState('');
  var [selected, setSelected] = useState(null),
    [detail, setDetail] = useState(null),
    [detailError, setDetailError] = useState('');
  var [saved, setSaved] = useState(() => {
    try {
      var v = JSON.parse(localStorage.getItem('charlie.desk.plans') || '[]');
      return Array.isArray(v) ? v : [];
    } catch {
      return [];
    }
  });
  var [planName, setPlanName] = useState('');
  var alive = useRef(true),
    request = useRef(0),
    launchLock = useRef(false);
  var fetchJson = async path => {
    var controller = new AbortController();
    var timer = setTimeout(() => controller.abort(), 20000);
    try {
      var res = await fetch(`${api}${path}`, {
        signal: controller.signal
      });
      if (!res.ok) throw new Error(res.status === 401 ? 'Sign in to load this data.' : `Request failed (${res.status})`);
      var result = await res.json();
      if (result.error) throw new Error(result.error);
      return result;
    } finally {
      clearTimeout(timer);
    }
  };
  var refresh = async () => {
    var id = ++request.current;
    var sources = [['runs', '/api/agents/results?limit=100', 'runs'], ['activities', '/api/analyst-activities/pending', 'activities'], ['failed', '/api/analyst-activities/failed?limit=100', 'activities'], ['analysts', '/api/analysts', 'analysts'], ['providers', '/api/agents/providers', null], ['capabilities', '/api/agents/capabilities', null]];
    var results = await Promise.allSettled(sources.map(([, path]) => fetchJson(path)));
    if (!alive.current || id !== request.current) return;
    var next = {},
      failed = [];
    results.forEach((r, i) => {
      var [key,, field] = sources[i];
      if (r.status === 'fulfilled') next[key] = field ? r.value[field] || [] : r.value;else if (key !== 'capabilities') failed.push(`${key}: ${r.reason.message}`);
    });
    setData(old => ({
      ...old,
      ...next
    }));
    setErrors(failed);
    setLoading(false);
    setUpdated(new Date());
  };
  useEffect(() => {
    alive.current = true;
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 30000);
    return () => {
      alive.current = false;
      clearInterval(timer);
      request.current++;
    };
  }, [api]);
  useEffect(() => {
    var models = data.providers[provider] || [];
    if (!models.includes(model)) setModel(models[0] || '');
  }, [provider, data.providers]);
  useEffect(() => {
    if (!selected) return;
    var current = true;
    setDetail(null);
    setDetailError('');
    fetchJson(`/api/agents/status/${encodeURIComponent(selected.id)}`).then(d => {
      if (current) setDetail(d);
    }).catch(e => {
      if (current) setDetailError(e.message);
    });
    return () => {
      current = false;
    };
  }, [selected]);
  var parsed = parseTickers(tickers),
    counts = runCounts(data.runs),
    statusChecks = runsNeedingStatusCheck(data.runs),
    queue = researchQueue(analyses, days),
    pending = data.activities.filter(a => a.status === 'pending_review');
  var max = Math.min(12, data.capabilities.maxBatchSize || 12),
    valid = parsed.tickers.length > 0 && parsed.tickers.length <= max && !parsed.invalid.length && !!model && !!date;
  var prepare = values => {
    setTickers(values.join(', '));
    setSection('launch');
    setReview(false);
    setMessage('');
  };
  var savePlan = () => {
    if (!planName.trim() || !valid) return;
    var plan = {
      name: planName.trim().slice(0, 60),
      tickers,
      provider,
      model,
      concurrency
    };
    var next = [plan, ...saved.filter(p => p.name !== plan.name)].slice(0, 12);
    try {
      localStorage.setItem('charlie.desk.plans', JSON.stringify(next));
      setSaved(next);
      setPlanName('');
      setMessage('Plan saved in this browser.');
    } catch {
      setMessage('This browser could not save the plan.');
    }
  };
  var launch = async () => {
    if (!valid || launchLock.current) return;
    launchLock.current = true;
    setBusy(true);
    setMessage('');
    try {
      var res = await fetch(`${api}/api/agents/batch-run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          tickers: parsed.tickers,
          date,
          provider,
          model,
          concurrency
        })
      });
      var result = await res.json();
      if (!res.ok) throw new Error(result.error || 'Could not start batch');
      setMessage(`Submitted ${result.count} company teams. Track their progress in Runs.`);
      setReview(false);
      setSection('runs');
      await refresh();
    } catch (e) {
      setMessage(`Submission could not be confirmed: ${e.message}. Check Runs before trying again.`);
      setSection('runs');
      await refresh();
    } finally {
      setBusy(false);
      launchLock.current = false;
    }
  };
  return /*#__PURE__*/React.createElement("div", {
    className: "workspace-page research-desk"
  }, /*#__PURE__*/React.createElement("div", {
    className: "desk-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "CHARLIE / RESEARCH OPERATIONS"), /*#__PURE__*/React.createElement("h1", null, "Your research desk."), /*#__PURE__*/React.createElement("p", {
    className: "workspace-lead"
  }, "Prioritize the next question. Coordinate the team. Review the evidence.")), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    onClick: () => setSection('launch')
  }, "Prepare a research batch \u2197")), /*#__PURE__*/React.createElement("div", {
    className: "desk-toolbar"
  }, /*#__PURE__*/React.createElement("nav", {
    "aria-label": "Research desk sections"
  }, [['overview', 'Overview'], ['collection', 'Collection'], ['history', 'Research history'], ['priorities', 'Portfolio priorities'], ['earnings', 'Earnings & evidence'], ['evidence', 'Evidence & changes'], ['inbox', 'Review inbox'], ['queue', 'Coverage queue'], ['launch', 'Batch planner'], ['runs', 'Runs'], ['playbooks', 'Playbooks']].map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    "aria-current": section === id ? 'page' : undefined,
    onClick: () => setSection(id)
  }, label))), /*#__PURE__*/React.createElement("button", {
    onClick: refresh,
    disabled: loading
  }, loading ? 'Loading…' : 'Refresh', updated && /*#__PURE__*/React.createElement("small", null, "Updated ", updated.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  })))), errors.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "workspace-error",
    role: "alert"
  }, "Some live data could not be refreshed. Previously loaded records may be out of date.", /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Connection details"), errors.map(e => /*#__PURE__*/React.createElement("p", {
    key: e
  }, e))), /*#__PURE__*/React.createElement("button", {
    onClick: refresh
  }, "Retry")), message && /*#__PURE__*/React.createElement("p", {
    className: "workspace-notice",
    role: "status"
  }, message), section === 'overview' && ['127.0.0.1', 'localhost'].includes(window.location.hostname) && /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "SOURCE OPERATIONS"), /*#__PURE__*/React.createElement("h2", null, "AlphaSense collection")), /*#__PURE__*/React.createElement("a", {
    className: "workspace-link-row",
    href: "http://127.0.0.1:8766/",
    target: "_blank",
    rel: "noopener noreferrer"
  }, "Open collection monitor \u2197")), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Manage ticker refresh schedules, verified originals, iCloud handoffs, and source restrictions. The monitor runs on this Mac.")), loading ? /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty",
    role: "status"
  }, "Loading research activity\u2026") : /*#__PURE__*/React.createElement(React.Fragment, null, section === 'history' && /*#__PURE__*/React.createElement(ResearchHistory, {
    api: api,
    onSection: setSection,
    onNavigate: onNavigate
  }), section === 'earnings' && /*#__PURE__*/React.createElement(EarningsWorkspace, {
    api: api,
    onRefresh: refresh,
    activities: [...data.activities, ...data.failed],
    onNavigate: onNavigate,
    onCompany: onCompany,
    renderHtml: renderRecapHtml || renderHtml
  }), section === 'evidence' && /*#__PURE__*/React.createElement(EvidenceWorkspace, {
    api: api,
    analyses: analyses,
    onCompany: onCompany
  }), section === 'overview' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "desk-metrics"
  }, [[counts.active, 'Active in latest 100 runs', 'runs'], [pending.length, 'Awaiting your review', 'inbox'], [queue.length, `Theses ${days}+ days old or undated`, 'queue'], [data.analysts.length, 'Sector analysts', 'playbooks']].map(([n, label, id]) => /*#__PURE__*/React.createElement("button", {
    key: label,
    onClick: () => setSection(id)
  }, /*#__PURE__*/React.createElement("strong", null, n), /*#__PURE__*/React.createElement("span", null, label), /*#__PURE__*/React.createElement("small", null, "Open \u2192")))), statusChecks.length > 0 && /*#__PURE__*/React.createElement("p", {
    className: "workspace-notice"
  }, statusChecks.length, " run(s) have been marked active for more than two hours. Their status may be stale. ", /*#__PURE__*/React.createElement("button", {
    className: "underline",
    onClick: () => setSection('runs')
  }, "Inspect runs \u2192")), /*#__PURE__*/React.createElement("div", {
    className: "desk-columns"
  }, /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, "The next research pass"), /*#__PURE__*/React.createElement("button", {
    onClick: () => setSection('queue')
  }, "View queue \u2197")), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Prioritized by thesis age. This identifies research maintenance, not investment attractiveness."), queue.slice(0, 5).map(a => /*#__PURE__*/React.createElement("div", {
    className: "desk-row",
    key: a.ticker
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => onCompany(a.ticker, 'portfolio')
  }, /*#__PURE__*/React.createElement("strong", null, a.ticker), /*#__PURE__*/React.createElement("span", null, a.companyName || a.company || 'Investment thesis')), /*#__PURE__*/React.createElement("small", null, a.age === null ? 'Undated' : `${a.age} days`), /*#__PURE__*/React.createElement("button", {
    onClick: () => prepare([a.ticker])
  }, "Research \u2192"))), !queue.length && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "No theses meet this age threshold.")), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel desk-team-flow"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "HOW THE TEAM WORKS"), /*#__PURE__*/React.createElement("h2", null, "Independent research. A challenged conclusion."), /*#__PURE__*/React.createElement("ol", null, /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("b", null, "01"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "Gather evidence"), /*#__PURE__*/React.createElement("p", null, "Market, fundamentals, news and sentiment analysts contribute specialized reports."))), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("b", null, "02"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "Debate the case"), /*#__PURE__*/React.createElement("p", null, "Bull and bear researchers challenge the conclusions before risk review."))), /*#__PURE__*/React.createElement("li", null, /*#__PURE__*/React.createElement("b", null, "03"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, "Review the decision"), /*#__PURE__*/React.createElement("p", null, "Read the combined output and compare it with your saved thesis.")))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    onClick: () => setSection('launch')
  }, "Coordinate a batch \u2192"))), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, "Analyst review inbox"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('analysts')
  }, "Open analyst team \u2197")), pending.slice(0, 6).map(a => /*#__PURE__*/React.createElement("div", {
    className: "desk-row",
    key: a.id
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, a.ticker || a.analystName || 'Research activity'), /*#__PURE__*/React.createElement("p", null, a.activityType?.replaceAll('_', ' '), " \xB7 ", a.analystName)), /*#__PURE__*/React.createElement(Status, {
    value: a.status
  }), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('analysts')
  }, "Review in inbox \u2192"))), !pending.length && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "No analyst output is waiting for review."))), section === 'inbox' && /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, "Analyst review inbox"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('analysts')
  }, "Open analyst controls \u2197")), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Read the proposed research before approving or revising it in Analyst team."), pending.map(a => /*#__PURE__*/React.createElement("details", {
    className: "desk-inbox-item",
    key: a.id
  }, /*#__PURE__*/React.createElement("summary", null, /*#__PURE__*/React.createElement("strong", null, a.ticker || a.analystName), /*#__PURE__*/React.createElement("span", null, a.activityType?.replaceAll('_', ' '), " \xB7 ", a.analystName), /*#__PURE__*/React.createElement("small", null, stamp(a.createdAt))), /*#__PURE__*/React.createElement("div", {
    className: "desk-inbox-report"
  }, /*#__PURE__*/React.createElement("h3", null, a.output ? 'Research output' : 'Proposed work'), a.output?.synthesisMarkdown ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Draft ready \xB7 ", a.output.fileCount || a.output.sourceFiles?.length || 0, " sources \xB7 Completed ", stamp(a.output.completedAt)), /*#__PURE__*/React.createElement(ReportContent, {
    value: a.output.synthesisMarkdown,
    renderHtml: renderRecapHtml || renderHtml
  }), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source documents and provenance"), /*#__PURE__*/React.createElement(ReportContent, {
    value: {
      sources: a.output.sourceFiles,
      provenance: a.output.sourceProvenance
    },
    renderHtml: renderHtml
  }))) : /*#__PURE__*/React.createElement(ReportContent, {
    value: a.output || a.input,
    renderHtml: renderHtml
  })), /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    onClick: () => onNavigate('analysts')
  }, "Review in analyst team \u2192"))), !pending.length && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "No analyst output is awaiting review.")), section === 'priorities' && /*#__PURE__*/React.createElement(PortfolioPriorities, {
    api: api,
    analyses: analyses,
    activities: [...data.activities, ...data.failed],
    onCompany: onCompany,
    onPlan: prepare
  }), section === 'collection' && /*#__PURE__*/React.createElement(CollectionCloud, {
    api: api,
    coverage: analyses
  }), section === 'queue' && /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, "Coverage maintenance"), /*#__PURE__*/React.createElement("button", {
    onClick: () => prepare(queue.slice(0, max).map(a => a.ticker)),
    disabled: !queue.length
  }, "Plan oldest ", Math.min(queue.length, max), " \u2192")), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Review threshold", /*#__PURE__*/React.createElement("select", {
    value: days,
    onChange: e => setDays(Number(e.target.value))
  }, [30, 60, 90, 180].map(d => /*#__PURE__*/React.createElement("option", {
    key: d,
    value: d
  }, d, " days")))), /*#__PURE__*/React.createElement("label", null, "Find a company", /*#__PURE__*/React.createElement("input", {
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Ticker or company"
  }))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Age is based on the saved thesis timestamp. Undated theses also appear here. Starting a batch creates new research; it does not overwrite the thesis."), queue.filter(a => `${a.ticker} ${a.companyName || a.company || ''}`.toLowerCase().includes(query.toLowerCase())).map(a => /*#__PURE__*/React.createElement("div", {
    className: "desk-row",
    key: a.ticker
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => onCompany(a.ticker, 'portfolio')
  }, /*#__PURE__*/React.createElement("strong", null, a.ticker), /*#__PURE__*/React.createElement("span", null, a.companyName || a.company)), /*#__PURE__*/React.createElement("small", null, a.age === null ? 'Undated' : `${a.age} days old`), /*#__PURE__*/React.createElement("button", {
    onClick: () => prepare([a.ticker])
  }, "Plan research \u2192"))), !queue.length && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "No theses meet this review threshold.")), section === 'launch' && /*#__PURE__*/React.createElement("div", {
    className: "desk-columns"
  }, /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "RESEARCH BATCH"), /*#__PURE__*/React.createElement("h2", null, "One brief. Multiple company teams."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Each company receives the existing multi-agent analysis. Results are saved as agent runs for your review."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy,
    className: "desk-form"
  }, /*#__PURE__*/React.createElement("label", null, "Companies", /*#__PURE__*/React.createElement("textarea", {
    "aria-label": "Batch tickers",
    value: tickers,
    onChange: e => {
      setTickers(e.target.value);
      setReview(false);
    },
    placeholder: "DE, CAT, ETN",
    rows: 3
  })), /*#__PURE__*/React.createElement("small", null, parsed.tickers.length, " / ", max, " companies \xB7 comma or space separated"), parsed.invalid.length > 0 && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "Unrecognized symbols: ", parsed.invalid.join(', ')), parsed.tickers.length > max && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "Limit this batch to ", max, " companies."), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Provider", /*#__PURE__*/React.createElement("select", {
    value: provider,
    onChange: e => {
      setProvider(e.target.value);
      setReview(false);
    }
  }, Object.keys(data.providers).map(p => /*#__PURE__*/React.createElement("option", {
    key: p
  }, p)))), /*#__PURE__*/React.createElement("label", null, "Analysis date", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: date,
    onChange: e => {
      setDate(e.target.value);
      setReview(false);
    }
  }))), /*#__PURE__*/React.createElement("label", null, "Model", /*#__PURE__*/React.createElement("select", {
    value: model,
    onChange: e => {
      setModel(e.target.value);
      setReview(false);
    }
  }, (data.providers[provider] || []).map(m => /*#__PURE__*/React.createElement("option", {
    key: m
  }, m)))), /*#__PURE__*/React.createElement("label", null, "Simultaneous company teams", /*#__PURE__*/React.createElement("select", {
    value: concurrency,
    onChange: e => {
      setConcurrency(Number(e.target.value));
      setReview(false);
    }
  }, Array.from({
    length: Math.min(3, data.capabilities.maxConcurrency || 1)
  }, (_, i) => /*#__PURE__*/React.createElement("option", {
    key: i + 1,
    value: i + 1
  }, i + 1, i === 0 ? ' · sequential' : '')))), data.capabilities.maxConcurrency === 1 && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "This backend currently supports sequential batches. The parallel-execution backend update unlocks up to three simultaneous teams."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: !valid,
    onClick: () => setReview(true)
  }, "Review batch \u2192")), review && /*#__PURE__*/React.createElement("section", {
    className: "desk-confirm",
    "aria-label": "Batch confirmation"
  }, /*#__PURE__*/React.createElement("h3", null, "Ready to start ", parsed.tickers.length, " company teams?"), /*#__PURE__*/React.createElement("p", null, parsed.tickers.join(' · ')), /*#__PURE__*/React.createElement("p", null, provider, " / ", model, /*#__PURE__*/React.createElement("br", null), date, " \xB7 ", concurrency, " at a time"), /*#__PURE__*/React.createElement("p", null, "Uses paid model APIs. Total cost varies by model and research depth. Outputs remain separate from your saved theses."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy || !valid,
    onClick: launch
  }, busy ? 'Submitting…' : 'Start research batch'), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setReview(false)
  }, "Keep editing"))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h2", null, "Reusable plans"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Save company groups and model preferences in this browser. Loading a plan never starts a run."), /*#__PURE__*/React.createElement("div", {
    className: "desk-form"
  }, /*#__PURE__*/React.createElement("label", null, "Plan name", /*#__PURE__*/React.createElement("input", {
    value: planName,
    onChange: e => setPlanName(e.target.value),
    placeholder: "Industrial coverage review"
  })), /*#__PURE__*/React.createElement("button", {
    className: "workspace-secondary",
    disabled: !valid || !planName.trim(),
    onClick: savePlan
  }, "Save current plan")), saved.map(p => /*#__PURE__*/React.createElement("button", {
    key: p.name,
    className: "desk-plan",
    onClick: () => {
      setTickers(p.tickers);
      setProvider(p.provider);
      setModel(p.model);
      setConcurrency(Math.min(p.concurrency || 1, data.capabilities.maxConcurrency || 1));
      setReview(false);
    }
  }, /*#__PURE__*/React.createElement("strong", null, p.name), /*#__PURE__*/React.createElement("small", null, p.tickers)))), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h2", null, "Execution contract"), /*#__PURE__*/React.createElement("ul", {
    className: "desk-contract"
  }, /*#__PURE__*/React.createElement("li", null, "One company per research team."), /*#__PURE__*/React.createElement("li", null, "Independent teams may run concurrently when supported by the backend."), /*#__PURE__*/React.createElement("li", null, "A failed team does not stop sibling teams."), /*#__PURE__*/React.createElement("li", null, "These are background jobs, not recurring schedules."), /*#__PURE__*/React.createElement("li", null, "Backend restarts can interrupt work; review run status before retrying."))))), section === 'runs' && /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, "Research runs"), /*#__PURE__*/React.createElement("span", null, "Latest 100 \xB7 updates every 30 seconds while visible")), data.runs.map(r => /*#__PURE__*/React.createElement("button", {
    className: "desk-run",
    key: r.id,
    onClick: () => setSelected(r)
  }, /*#__PURE__*/React.createElement("strong", null, r.ticker), /*#__PURE__*/React.createElement("span", null, r.model, /*#__PURE__*/React.createElement("small", null, stamp(r.createdAt))), /*#__PURE__*/React.createElement(Status, {
    value: r.status
  }), /*#__PURE__*/React.createElement("span", null, "Inspect \u2192"))), !data.runs.length && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "No agent runs have been recorded. Prepare your first batch when you are ready.")), section === 'playbooks' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "workspace-create-grid"
  }, PLAYBOOKS.map(p => /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel",
    key: p.id
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, p.tag), /*#__PURE__*/React.createElement("h2", null, p.name), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, p.description), /*#__PURE__*/React.createElement("ol", {
    className: "desk-contract"
  }, p.steps.map(s => /*#__PURE__*/React.createElement("li", {
    key: s
  }, s))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    onClick: () => p.id === 'refresh' ? setSection('launch') : onNavigate(p.view)
  }, "Open workflow \u2192")))), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h2", null, "Your sector analysts"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Specialized coverage and research instructions are managed in Analyst team."), /*#__PURE__*/React.createElement("div", {
    className: "desk-roster"
  }, data.analysts.map(a => /*#__PURE__*/React.createElement("button", {
    key: a.id,
    onClick: () => onNavigate('analysts')
  }, /*#__PURE__*/React.createElement("strong", null, a.name), /*#__PURE__*/React.createElement("span", null, a.sector, " \xB7 ", a.coverageTickers?.length || 0, " companies"), /*#__PURE__*/React.createElement("small", null, a.pendingCount || 0, " awaiting review \xB7 ", a.autoMode?.enabled ? 'Auto mode enabled' : 'Manual review')))), !data.analysts.length && /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    onClick: () => onNavigate('analysts')
  }, "Configure your analyst team \u2192")))), selected && /*#__PURE__*/React.createElement("section", {
    className: "desk-detail",
    role: "region",
    "aria-label": `${selected.ticker} run details`
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, selected.ticker, " research run"), /*#__PURE__*/React.createElement("button", {
    autoFocus: true,
    onClick: () => setSelected(null)
  }, "Close")), /*#__PURE__*/React.createElement(Status, {
    value: detail?.status || selected.status
  }), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Saved ", stamp(selected.createdAt), " \xB7 Analysis date ", selected.analysisDate, " \xB7 ", selected.model), detailError ? /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, detailError) : !detail ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading run\u2026") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h3", null, "Decision"), /*#__PURE__*/React.createElement(ReportContent, {
    value: detail.decision || 'No decision yet.',
    renderHtml: renderHtml
  }), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Research reports"), /*#__PURE__*/React.createElement(ReportContent, {
    value: detail.report,
    renderHtml: renderHtml
  })), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Execution log"), /*#__PURE__*/React.createElement("pre", null, (detail.newLogs || []).map(l => typeof l === 'string' ? l : JSON.stringify(l)).join('\n') || 'No log entries.')), /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    onClick: () => {
      setSelected(null);
      onCompany(selected.ticker, 'portfolio');
    }
  }, "Compare with saved thesis \u2192"))));
}