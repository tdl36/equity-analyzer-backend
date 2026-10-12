import * as React from 'react';
import { activityRows, activityAge, collectionHealth, workLane, nextAction, workOverview, focusRows, activityDestination } from './activity-model.mjs';
function AnalystRunInspection({
  api,
  row
}) {
  var [state, setState] = React.useState({
      loading: true
    }),
    [revision, setRevision] = React.useState(0);
  React.useEffect(() => {
    var current = true;
    var controller = new AbortController(),
      timer = setTimeout(() => controller.abort(), 15000);
    setState({
      loading: true
    });
    (async () => {
      try {
        var response = await fetch(`${api}/api/activity/analyst-run/${encodeURIComponent(row.jobId)}`, {
          signal: controller.signal
        });
        if (!response.ok) throw new Error('Saved run unavailable');
        var _data = await response.json();
        if (_data.id !== row.jobId || _data.ticker !== row.ticker) throw new Error('Saved run identity mismatch');
        if (current) setState({
          data: _data
        });
      } catch (e) {
        if (current) setState({
          error: 'This exact saved run could not be loaded. Refresh to retry this read; no research has been started.'
        });
      } finally {
        clearTimeout(timer);
      }
    })();
    return () => {
      current = false;
      clearTimeout(timer);
      controller.abort();
    };
  }, [api, row.jobId, row.ticker, revision]);
  var [saving, setSaving] = React.useState(false),
    [saveError, setSaveError] = React.useState('');
  var recordUnresolved = async () => {
    if (saving) return;
    setSaving(true);
    setSaveError('');
    var controller = new AbortController(),
      timer = setTimeout(() => controller.abort(), 15000);
    try {
      var r = await fetch(`${api}/api/activity/analyst-run/${encodeURIComponent(row.jobId)}/record-unresolved`, {
        method: 'POST',
        signal: controller.signal
      });
      if (!r.ok) throw new Error('Unable to record disposition');
      setRevision(n => n + 1);
    } catch (e) {
      setSaveError('Could not confirm the saved disposition. Refresh the saved state before trying again; no research was started.');
    } finally {
      clearTimeout(timer);
      setSaving(false);
    }
  };
  var data = state.data;
  return /*#__PURE__*/React.createElement("section", {
    className: "activity-run-inspection",
    "aria-label": `Saved analyst run for ${row.ticker}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "activity-row-title"
  }, /*#__PURE__*/React.createElement("strong", null, "Saved run \xB7 ", row.ticker), /*#__PURE__*/React.createElement("button", {
    disabled: state.loading,
    onClick: () => setRevision(n => n + 1)
  }, "Refresh saved state")), state.loading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading saved steps\u2026"), state.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, state.error), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "Recorded status: ", /*#__PURE__*/React.createElement("strong", null, data.status), " \xB7 ", data.model, " \xB7 Created ", activityAge(data.createdAt)), /*#__PURE__*/React.createElement("p", null, data.hasReport ? 'Report text is saved.' : 'No report text is saved.', " ", data.completedAt ? `Completion recorded ${activityAge(data.completedAt)}.` : 'No completion is recorded.'), /*#__PURE__*/React.createElement("p", {
    className: "activity-warning"
  }, "These are saved records, not confirmation that a worker or provider request is running. Inspect the outcome before starting another paid run."), data.workerPresence && /*#__PURE__*/React.createElement("p", null, "Worker reporting: ", /*#__PURE__*/React.createElement("strong", null, {
    reporting: 'recent heartbeat',
    exited: 'worker exited',
    stale: 'heartbeat expired',
    unknown: 'not established',
    unavailable: 'check unavailable'
  }[data.workerPresence.state] || 'not established'), data.workerPresence.lastSeen ? ` · ${activityAge(data.workerPresence.lastSeen)}` : '', ". This describes the worker reporter, not provider completion or permission to retry."), saveError && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, saveError), data.legacyDispositionAvailable && /*#__PURE__*/React.createElement("div", {
    className: "activity-warning"
  }, /*#__PURE__*/React.createElement("p", null, "This legacy run has no recent saved progress or output. Record it as needing attention while preserving its unknown provider outcome. This does not cancel or restart research."), /*#__PURE__*/React.createElement("button", {
    disabled: saving,
    onClick: recordUnresolved
  }, saving ? 'Recording…' : 'Record unresolved outcome')), data.executionEvidence && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Recovery evidence"), /*#__PURE__*/React.createElement("p", null, "Saved run ID: ", /*#__PURE__*/React.createElement("span", {
    style: {
      overflowWrap: 'anywhere'
    }
  }, data.id)), /*#__PURE__*/React.createElement("p", null, data.executionEvidence.enrolled ? 'This run has a saved execution enrollment.' : 'This run has no execution enrollment record; absence does not prove it never ran.', " Recorded reservations: ", data.executionEvidence.reservationCount, "."), data.executionEvidence.reservationIds.map(id => /*#__PURE__*/React.createElement("p", {
    key: id,
    style: {
      overflowWrap: 'anywhere'
    }
  }, "Execution reference: ", id)), data.executionEvidence.reservationIdsTruncated && /*#__PURE__*/React.createElement("p", null, "Showing the first five execution references."), data.executionEvidence.modelCalls && /*#__PURE__*/React.createElement("p", null, "Graph model calls recorded: ", data.executionEvidence.modelCalls.recordedCalls, "; responses returned: ", data.executionEvidence.modelCalls.returned, "; errors recorded: ", data.executionEvidence.modelCalls.errors, "; starts without an outcome: ", data.executionEvidence.modelCalls.unresolved, ". Coverage is limited to graph callbacks. Errors or missing records do not establish billing, cancellation or permission to retry."), data.executionEvidence.modelCalls?.responseIds?.length > 0 && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", null, "Saved provider response references for reconciliation:"), data.executionEvidence.modelCalls.responseIds.map(id => /*#__PURE__*/React.createElement("p", {
    key: id,
    style: {
      overflowWrap: 'anywhere',
      fontFamily: 'monospace'
    }
  }, id)), data.executionEvidence.modelCalls.responseIdsTruncated && /*#__PURE__*/React.createElement("p", null, "Showing the first 20 response references."), /*#__PURE__*/React.createElement("p", null, "These identify returned responses, not billing receipts or proof that the whole job completed.")), /*#__PURE__*/React.createElement("p", null, "Provider outcome is unverified. Execution references identify Charlie reservations, not provider request IDs. Reconcile backend execution and provider records before authorizing any new attempt.")), data.truncated && /*#__PURE__*/React.createElement("p", null, "Showing the latest 100 of ", data.totalLogs, " saved entries."), /*#__PURE__*/React.createElement("ol", null, data.logs.map((entry, index) => /*#__PURE__*/React.createElement("li", {
    key: index
  }, /*#__PURE__*/React.createElement("small", null, entry.time || 'Time unavailable', " \xB7 ", entry.agent || 'Saved step'), /*#__PURE__*/React.createElement("p", null, entry.message)))), !data.logs.length && /*#__PURE__*/React.createElement("p", null, "No saved steps are available."), /*#__PURE__*/React.createElement("small", null, "Run ", data.id, " \xB7 Retrieved ", activityAge(data.asOf))));
}
export function ActivityWorkspace({
  api,
  onNavigate,
  compact = false
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [filter, setFilter] = React.useState('focus'),
    [query, setQuery] = React.useState(''),
    [limit, setLimit] = React.useState(20);
  var [inspecting, setInspecting] = React.useState('');
  var [loading, setLoading] = React.useState(false);
  var mounted = React.useRef(false),
    pending = React.useRef(false),
    controller = React.useRef(null);
  var load = async () => {
    if (pending.current) return;
    pending.current = true;
    setLoading(true);
    var c = new AbortController();
    controller.current = c;
    var timeout = setTimeout(() => c.abort(), 15000);
    try {
      var r = await fetch(`${api}/api/activity`, {
        signal: c.signal
      });
      if (!r.ok) throw new Error('Activity could not be refreshed.');
      var d = await r.json();
      if (mounted.current) {
        setData(d);
        setError('');
      }
    } catch (e) {
      if (mounted.current) setError('Activity could not be refreshed. Displayed information may be out of date.');
    } finally {
      clearTimeout(timeout);
      pending.current = false;
      if (mounted.current) setLoading(false);
    }
  };
  React.useEffect(() => {
    mounted.current = true;
    load();
    var timer = setInterval(() => {
      if (!document.hidden) load();
    }, 15000);
    var visible = () => {
      if (!document.hidden) load();
    };
    document.addEventListener('visibilitychange', visible);
    return () => {
      mounted.current = false;
      clearInterval(timer);
      controller.current?.abort();
      document.removeEventListener('visibilitychange', visible);
    };
  }, [api]);
  React.useEffect(() => setLimit(20), [filter, query]);
  var matching = activityRows(data, filter, query);
  var rows = compact ? focusRows(data).slice(0, 4) : matching.slice(0, limit);
  var counts = workOverview(data);
  counts.focus = counts.blocked + counts.review;
  var lanes = {
    blocked: 'Check progress',
    review: 'Your review',
    processing: 'Processing',
    waiting: 'Waiting',
    recent: 'Finished',
    older: 'Older unfinished'
  };
  return /*#__PURE__*/React.createElement("section", {
    className: compact ? 'workspace-panel activity-compact' : 'workspace-page activity-page',
    "aria-label": compact ? 'Work in progress' : 'Activity'
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading activity-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "ACROSS YOUR RESEARCH"), compact ? /*#__PURE__*/React.createElement("h2", null, "Work in progress") : /*#__PURE__*/React.createElement("h1", null, "Activity."), /*#__PURE__*/React.createElement("p", null, compact ? 'Decisions and stalled work first, then progress across your companies.' : 'See what Charlie is working on, what is waiting, and what is ready to review.')), /*#__PURE__*/React.createElement("button", {
    onClick: compact ? () => onNavigate('activity') : load,
    disabled: !compact && loading
  }, compact ? 'View all activity ↗' : loading ? 'Refreshing…' : 'Refresh')), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "activity-warning"
  }, error), !!data?.unavailable?.length && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "activity-warning"
  }, "Some activity is unavailable: ", data.unavailable.join(', '), ". This is a partial snapshot."), !!data?.truncated?.length && /*#__PURE__*/React.createElement("p", {
    className: "activity-warning"
  }, "Showing the first 200 unfinished items in ", data.truncated.join(', '), ". Open that workspace for more."), data && /*#__PURE__*/React.createElement("div", {
    className: "activity-health",
    "aria-label": "Collection delivery health"
  }, collectionHealth(data).map(h => /*#__PURE__*/React.createElement("div", {
    key: h.key,
    className: h.ok ? '' : 'needs-check'
  }, /*#__PURE__*/React.createElement("small", null, h.label), /*#__PURE__*/React.createElement("strong", null, h.value), !compact && /*#__PURE__*/React.createElement("p", null, h.detail)))), !compact && data?.meetingAgenda && /*#__PURE__*/React.createElement("details", {
    className: "activity-agenda",
    open: data.meetingAgenda.items.length > 0
  }, /*#__PURE__*/React.createElement("summary", null, "Upcoming meeting preparation \xB7 ", data.meetingAgenda.items.length, data.meetingAgenda.truncated ? '+' : ''), /*#__PURE__*/React.createElement("p", null, "Saved meeting dates from ", data.meetingAgenda.from, " through ", data.meetingAgenda.through, " (New York dates). This is your Charlie preparation list, not a synced calendar or attendance record."), data.meetingAgenda.items.length ? data.meetingAgenda.items.map(m => /*#__PURE__*/React.createElement("article", {
    className: "activity-row",
    key: m.id
  }, /*#__PURE__*/React.createElement("div", {
    className: "activity-company"
  }, /*#__PURE__*/React.createElement("strong", null, m.ticker), /*#__PURE__*/React.createElement("small", null, m.date)), /*#__PURE__*/React.createElement("div", {
    className: "activity-detail"
  }, /*#__PURE__*/React.createElement("strong", null, m.prepared ? 'Question pack saved' : 'Preparation needed'), /*#__PURE__*/React.createElement("p", null, m.type || 'Meeting', " \xB7 ", m.prepared ? 'Review the questions and their factual premises before the meeting.' : 'Open the meeting to add sources or prepare questions.')), /*#__PURE__*/React.createElement("button", {
    className: "activity-open",
    onClick: () => onNavigate('meetingprep', m.ticker, '', 'meeting:' + m.id)
  }, "Open meeting \u2197", /*#__PURE__*/React.createElement("span", {
    className: "sr-only"
  }, " ", m.ticker, " ", m.date)))) : /*#__PURE__*/React.createElement("p", null, "No dated meetings in this seven-day window. Undated and other meetings remain in Meeting Preparation."), data.meetingAgenda.truncated && /*#__PURE__*/React.createElement("p", null, "Showing the earliest 50. Open Meeting Preparation for the full list.")), !compact && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "activity-counters",
    "aria-label": "Activity filters"
  }, [['focus', 'Needs you'], ['processing', 'Processing'], ['waiting', 'Waiting'], ['recent', 'Finished'], ['older', 'Older unfinished']].map(([key, label]) => /*#__PURE__*/React.createElement("button", {
    key: key,
    "aria-pressed": filter === key,
    onClick: () => setFilter(key)
  }, /*#__PURE__*/React.createElement("strong", null, data ? counts[key] : '—'), /*#__PURE__*/React.createElement("span", null, label)))), /*#__PURE__*/React.createElement("div", {
    className: "activity-toolbar"
  }, /*#__PURE__*/React.createElement("label", null, "Find work ", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Find work",
    placeholder: "Ticker or workflow",
    value: query,
    onChange: e => setQuery(e.target.value)
  })))), !data && !error && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading activity\u2026"), data && !rows.length && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, query ? 'No matching work.' : compact ? 'No current unfinished work. Older unresolved records remain in Activity.' : filter === 'focus' ? 'No current decisions or stalled jobs in this snapshot.' : filter === 'processing' ? 'No jobs currently reporting processing.' : filter === 'waiting' ? 'No jobs currently waiting.' : filter === 'older' ? 'No older unfinished records in this snapshot.' : 'No recently finished work in the last 24 hours.'), /*#__PURE__*/React.createElement("div", {
    className: "activity-list"
  }, rows.map(r => /*#__PURE__*/React.createElement("article", {
    className: 'activity-row activity-' + r.bucket,
    key: r.id
  }, /*#__PURE__*/React.createElement("div", {
    className: "activity-company"
  }, /*#__PURE__*/React.createElement("strong", null, r.ticker || 'GENERAL'), /*#__PURE__*/React.createElement("small", null, r.title)), /*#__PURE__*/React.createElement("div", {
    className: "activity-detail"
  }, /*#__PURE__*/React.createElement("div", {
    className: "activity-row-title"
  }, /*#__PURE__*/React.createElement("strong", null, r.error && r.step === r.error ? 'Collection needs attention' : r.step), /*#__PURE__*/React.createElement("span", {
    className: "activity-status"
  }, lanes[workLane(r)])), r.stages != null && /*#__PURE__*/React.createElement("p", null, r.stages, "/12 research stages saved"), r.warning && /*#__PURE__*/React.createElement("p", {
    className: "activity-warning"
  }, r.warning), r.error && (r.error.length > 200 ? /*#__PURE__*/React.createElement("details", {
    className: "activity-warning"
  }, /*#__PURE__*/React.createElement("summary", null, r.error.slice(0, 180), "\u2026 ", /*#__PURE__*/React.createElement("span", null, "View full issue")), /*#__PURE__*/React.createElement("p", null, r.error)) : /*#__PURE__*/React.createElement("p", {
    className: "activity-warning"
  }, r.error)), r.stale && /*#__PURE__*/React.createElement("p", {
    className: "activity-warning"
  }, "No saved progress for ", r.quietMinutes == null ? 'an unknown period' : r.quietMinutes + ' minutes', ". Open the job to inspect what is waiting."), /*#__PURE__*/React.createElement("p", {
    className: "activity-next"
  }, /*#__PURE__*/React.createElement("strong", null, "Next:"), " ", nextAction(r)), /*#__PURE__*/React.createElement("small", null, "Updated ", activityAge(r.updatedAt), " \xB7 Started ", activityAge(r.createdAt))), r.workflow === 'agent_runs' ? /*#__PURE__*/React.createElement("button", {
    className: "activity-open",
    "aria-expanded": inspecting === r.jobId,
    onClick: () => setInspecting(inspecting === r.jobId ? '' : r.jobId)
  }, inspecting === r.jobId ? 'Close saved run' : 'Inspect saved run', /*#__PURE__*/React.createElement("span", {
    className: "sr-only"
  }, " ", r.ticker)) : /*#__PURE__*/React.createElement("button", {
    className: "activity-open",
    onClick: () => {
      var d = activityDestination(r);
      onNavigate(d.view, r.ticker, d.section, d.work);
    }
  }, "Open work \u2197", /*#__PURE__*/React.createElement("span", {
    className: "sr-only"
  }, " ", r.ticker, " ", r.title)), r.workflow === 'agent_runs' && inspecting === r.jobId && /*#__PURE__*/React.createElement(AnalystRunInspection, {
    api: api,
    row: r
  })))), !compact && matching.length > limit && /*#__PURE__*/React.createElement("button", {
    className: "activity-open",
    onClick: () => setLimit(n => n + 20)
  }, "Show more \xB7 ", matching.length - limit, " remaining"), data && /*#__PURE__*/React.createElement("p", {
    className: "activity-foot"
  }, "Snapshot ", activityAge(data.asOf), " \xB7 Refreshes every 15 seconds while visible.", !compact && /*#__PURE__*/React.createElement(React.Fragment, null, " ", data.scope, " Mac collection report: ", activityAge(data.collectionLastSeen), ".")));
}