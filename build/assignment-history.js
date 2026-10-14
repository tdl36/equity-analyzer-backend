import * as React from 'react';
import { historyPage, appendHistory } from './assignment-history-model.mjs';
import { activityAge } from './activity-model.mjs';
import { routeHash } from './workspace-model.mjs';
export function AssignmentHistory({
  api
}) {
  var [open, setOpen] = React.useState(false),
    [query, setQuery] = React.useState(''),
    [filter, setFilter] = React.useState('');
  var [state, setState] = React.useState({
    rows: [],
    cursor: null,
    loaded: false,
    busy: false
  });
  var generation = React.useRef(0),
    active = React.useRef(null),
    locked = React.useRef(false);
  async function load(more = false) {
    if (locked.current) return;
    locked.current = true;
    var token = ++generation.current,
      c = new AbortController();
    active.current = c;
    var cursor = more ? state.cursor : null;
    setState(s => ({
      ...s,
      busy: true,
      error: ''
    }));
    try {
      var params = new URLSearchParams();
      if (filter) params.set('ticker', filter);
      if (cursor) params.set('cursor', cursor);
      var response = await fetch(`${api}/api/research/assignment-history?${params}`, {
        signal: AbortSignal.any([c.signal, AbortSignal.timeout(30000)])
      });
      var data = await response.json();
      if (!response.ok) throw Error(data.error || 'Saved assignments are unavailable.');
      var page = historyPage(data, filter);
      if (token === generation.current) setState(s => ({
        rows: more ? appendHistory(s.rows, page.rows) : page.rows,
        cursor: page.cursor,
        loaded: true,
        busy: false
      }));
    } catch (error) {
      if (token === generation.current) setState(s => ({
        ...s,
        busy: false,
        error: 'Could not load saved assignments. Your loaded rows remain available. Retry this read.'
      }));
    } finally {
      if (token === generation.current) locked.current = false;
    }
  }
  React.useEffect(() => {
    generation.current++;
    active.current?.abort();
    locked.current = false;
    setState({
      rows: [],
      cursor: null,
      loaded: false,
      busy: false
    });
    if (open) load();
    return () => {
      generation.current++;
      active.current?.abort();
      locked.current = false;
    };
  }, [api, open, filter]);
  var statuses = {
    complete: 'Finished · review saved outputs',
    completed: 'Finished · review saved outputs',
    attention: 'Needs attention',
    running: 'In progress',
    queued: 'Waiting',
    cancelled: 'Stopped',
    stopped: 'Stopped'
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "activity-archive",
    "aria-label": "Saved assignment archive"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setOpen(v => !v),
    "aria-expanded": open
  }, open ? 'Close assignment archive' : 'Find older assignments'), open && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h3", null, "Saved assignments"), /*#__PURE__*/React.createElement("p", null, "Browse across dates and companies. Finished work still requires your review."), /*#__PURE__*/React.createElement("form", {
    className: "activity-toolbar",
    onSubmit: e => {
      e.preventDefault();
      var value = query.trim().toUpperCase();
      if (value === filter) load();else setFilter(value);
    }
  }, /*#__PURE__*/React.createElement("label", null, "Company ticker ", /*#__PURE__*/React.createElement("input", {
    value: query,
    maxLength: 30,
    onChange: e => setQuery(e.target.value),
    placeholder: "All companies",
    "aria-label": "Archive company ticker",
    pattern: "[A-Za-z0-9.\\-]*"
  })), /*#__PURE__*/React.createElement("button", {
    type: "submit",
    disabled: state.busy
  }, "Find assignments")), state.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, state.error), state.busy && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading saved assignments\u2026"), state.loaded && !state.rows.length && /*#__PURE__*/React.createElement("p", null, "No saved assignments match this company filter."), state.rows.map(row => /*#__PURE__*/React.createElement("article", {
    className: "activity-row",
    key: row.id
  }, /*#__PURE__*/React.createElement("div", {
    className: "activity-company"
  }, /*#__PURE__*/React.createElement("strong", null, row.ticker), /*#__PURE__*/React.createElement("small", null, statuses[row.status] || 'Inspect saved status')), /*#__PURE__*/React.createElement("div", {
    className: "activity-detail"
  }, /*#__PURE__*/React.createElement("small", null, "Created ", activityAge(row.created_at), " \xB7 Updated ", activityAge(row.updated_at))), /*#__PURE__*/React.createElement("a", {
    className: "activity-open",
    href: routeHash('stockanalysis', row.ticker, 'stock-analysis', 'assignment:' + row.id)
  }, "Open saved assignment \u2197", /*#__PURE__*/React.createElement("span", {
    className: "sr-only"
  }, " ", row.ticker)))), state.cursor && /*#__PURE__*/React.createElement("button", {
    disabled: state.busy,
    onClick: () => load(true)
  }, state.error ? 'Retry older assignments' : 'Load older assignments'), state.error && !state.cursor && /*#__PURE__*/React.createElement("button", {
    disabled: state.busy,
    onClick: () => load()
  }, "Retry archive")));
}