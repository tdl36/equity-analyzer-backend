import * as React from 'react';
import { activityRows, activityAge, collectionHealth, workLane, nextAction, workOverview, focusRows, activityDestination } from './activity-model.mjs';
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
  }, "Showing the first 200 unfinished jobs in ", data.truncated.join(', '), ". Open that workspace for more."), data && /*#__PURE__*/React.createElement("div", {
    className: "activity-health",
    "aria-label": "Collection delivery health"
  }, collectionHealth(data).map(h => /*#__PURE__*/React.createElement("div", {
    key: h.key,
    className: h.ok ? '' : 'needs-check'
  }, /*#__PURE__*/React.createElement("small", null, h.label), /*#__PURE__*/React.createElement("strong", null, h.value), !compact && /*#__PURE__*/React.createElement("p", null, h.detail)))), !compact && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
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
  }, /*#__PURE__*/React.createElement("strong", null, "Next:"), " ", nextAction(r)), /*#__PURE__*/React.createElement("small", null, "Updated ", activityAge(r.updatedAt), " \xB7 Started ", activityAge(r.createdAt))), /*#__PURE__*/React.createElement("button", {
    className: "activity-open",
    onClick: () => {
      var d = activityDestination(r);
      onNavigate(d.view, r.ticker, d.section, d.work);
    }
  }, "Open work \u2197", /*#__PURE__*/React.createElement("span", {
    className: "sr-only"
  }, " ", r.ticker, " ", r.title))))), !compact && matching.length > limit && /*#__PURE__*/React.createElement("button", {
    className: "activity-open",
    onClick: () => setLimit(n => n + 20)
  }, "Show more \xB7 ", matching.length - limit, " remaining"), data && /*#__PURE__*/React.createElement("p", {
    className: "activity-foot"
  }, "Snapshot ", activityAge(data.asOf), " \xB7 Refreshes every 15 seconds while visible.", !compact && /*#__PURE__*/React.createElement(React.Fragment, null, " ", data.scope, " Mac collection report: ", activityAge(data.collectionLastSeen), ".")));
}