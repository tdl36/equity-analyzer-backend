import * as React from 'react';
import { AssignmentWorkspace } from './assignment-workspace';
import { workItems } from './my-work-model.mjs';
export function MyWork({
  api,
  activities,
  onNavigate,
  onSection,
  renderHtml
}) {
  var [meetings, setMeetings] = React.useState([]),
    [commands, setCommands] = React.useState([]),
    [error, setError] = React.useState(''),
    [filter, setFilter] = React.useState('all'),
    [query, setQuery] = React.useState('');
  React.useEffect(() => {
    var alive = true;
    var load = async () => {
      try {
        var values = await Promise.all(['/api/research/meeting-commands', '/api/research/commands'].map(async path => {
          var r = await fetch(api + path, {
            signal: AbortSignal.timeout(20000)
          });
          if (!r.ok) throw Error('Could not refresh assignments. Last loaded results may be stale.');
          return r.json();
        }));
        if (alive) {
          setMeetings(values[0].jobs || []);
          setCommands(values[1].jobs || []);
          setError('');
        }
      } catch (e) {
        if (alive) setError(e.message);
      }
    };
    load();
    var timer = setInterval(() => {
      if (!document.hidden) load();
    }, 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [api]);
  var items = workItems(meetings, commands, activities),
    labels = {
      all: 'All work',
      needs_me: 'Needs me',
      running: 'Running / queued',
      ready: 'Ready',
      failed: 'Failed',
      closed: 'Cancelled'
    };
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "MY WORK"), /*#__PURE__*/React.createElement("h2", null, "Assignments, answers and next actions.")), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    onClick: () => onSection('command')
  }, "New assignment \u2192")), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Recent meeting packs, research commands and analyst inbox items. Saved history remains in Research history. Updates while this page is open."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Find work", /*#__PURE__*/React.createElement("input", {
    type: "search",
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Ticker or task"
  })), /*#__PURE__*/React.createElement("label", null, "Status", /*#__PURE__*/React.createElement("select", {
    value: filter,
    onChange: e => setFilter(e.target.value)
  }, Object.entries(labels).map(([id, label]) => /*#__PURE__*/React.createElement("option", {
    key: id,
    value: id
  }, label, " \xB7 ", id === 'all' ? items.length : items.filter(i => i.bucket === id).length))))), /*#__PURE__*/React.createElement("div", {
    className: "meeting-pack-list"
  }, items.filter(i => (filter === 'all' || i.bucket === filter) && `${i.ticker} ${i.kind} ${i.activity?.input?.topic || ''}`.toLowerCase().includes(query.toLowerCase())).map(i => /*#__PURE__*/React.createElement("article", {
    key: i.kind + i.id
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h3", null, i.ticker, " \xB7 ", i.kind), /*#__PURE__*/React.createElement("span", {
    className: "desk-status"
  }, labels[i.bucket])), /*#__PURE__*/React.createElement("p", null, i.status), i.activity?.input?.topic && /*#__PURE__*/React.createElement("p", null, i.activity.input.topic), i.error && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error"
  }, i.error), i.job?.options?.meetingPrep && /*#__PURE__*/React.createElement("p", null, i.job.options.meetingPrep.format?.replaceAll('_', ' ') || 'Conference', " \xB7 ", i.job.options.meetingPrep.meetingDate), i.job && /*#__PURE__*/React.createElement(AssignmentWorkspace, {
    api: api,
    id: i.id,
    onNavigate: onNavigate,
    onSection: onSection
  }), i.job?.meeting_id && /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      window.dispatchEvent(new CustomEvent('charlie-open-meeting', {
        detail: {
          id: Number(i.job.meeting_id)
        }
      }));
      onNavigate('meetingprep');
    }
  }, "Open question pack \u2192"), i.activity && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Read recap and evidence"), i.activity.output?.synthesisMarkdown ? renderHtml ? /*#__PURE__*/React.createElement("div", {
    className: "desk-report-prose",
    dangerouslySetInnerHTML: {
      __html: renderHtml(i.activity.output.synthesisMarkdown)
    }
  }) : /*#__PURE__*/React.createElement("pre", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, i.activity.output.synthesisMarkdown) : /*#__PURE__*/React.createElement("p", null, "No completed recap is attached yet. Open Analyst inbox for this item\u2019s full details.")), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('analysts')
  }, "Review in Analyst inbox \u2192")), (i.bucket === 'failed' || i.bucket === 'needs_me') && i.job && /*#__PURE__*/React.createElement("button", {
    onClick: () => onSection(i.kind === 'Meeting pack' ? 'command' : 'collection')
  }, "Resolve / retry \u2192")))), !items.length && !error && /*#__PURE__*/React.createElement("p", null, "No recent assignments or inbox items loaded."));
}