import * as React from 'react';
import { questionSelection } from './research-meeting-model.mjs';
export function ResearchMeetingHandoff({
  api,
  runId,
  onMeeting
}) {
  var [pack, setPack] = React.useState(null),
    [indices, setIndices] = React.useState([]),
    [when, setWhen] = React.useState(''),
    [kind, setKind] = React.useState('management'),
    [message, setMessage] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [saved, setSaved] = React.useState(null),
    [reload, setReload] = React.useState(0);
  var live = React.useRef(true),
    locked = React.useRef(false);
  React.useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  React.useEffect(() => {
    var active = true;
    setPack(null);
    setIndices([]);
    setSaved(null);
    setMessage('');
    fetch(`${api}/api/research/stock-analysis-run/${runId}/meeting-prep`, {
      signal: AbortSignal.timeout(20000)
    }).then(async r => {
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Questions could not be loaded.');
      return d;
    }).then(d => {
      if (active) setPack(d);
    }).catch(e => {
      if (active) setMessage(e.message);
    });
    return () => {
      active = false;
    };
  }, [api, runId, reload]);
  var save = async () => {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setMessage('');
    try {
      var body = questionSelection(pack, indices, when, kind);
      var r = await fetch(`${api}/api/research/stock-analysis-run/${runId}/meeting-prep`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(30000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Meeting could not be saved.');
      if (live.current) {
        setSaved(d);
        setMessage(d.reused ? 'This selection was already saved. Open the existing meeting below.' : 'Selected questions saved to Meeting Prep.');
      }
    } catch (e) {
      if (live.current) setMessage(e.message + ' Your selection is retained. Retry the same selection to recover an uncertain save without creating a duplicate.');
    } finally {
      locked.current = false;
      if (live.current) setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("details", {
    className: "sa-prepare"
  }, /*#__PURE__*/React.createElement("summary", null, "Use these questions in a meeting"), /*#__PURE__*/React.createElement("p", null, "Select the questions you want to ask and choose a date. Charlie copies their exact wording into Meeting Prep, where you can capture answers. Selected questions become must-ask. No additional research is run."), /*#__PURE__*/React.createElement("p", null, "Questions can contain unverified premises. Review them against the linked report and originals before the meeting."), !pack && !message && /*#__PURE__*/React.createElement("p", null, "Loading saved questions\u2026"), !pack && message && /*#__PURE__*/React.createElement("button", {
    onClick: () => setReload(x => x + 1)
  }, "Reload questions"), pack && /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy
  }, /*#__PURE__*/React.createElement("div", null, pack.questions.map(q => /*#__PURE__*/React.createElement("label", {
    className: "sa-check",
    key: q.index
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: indices.includes(q.index),
    onChange: e => {
      setSaved(null);
      setIndices(old => e.target.checked ? [...old, q.index] : old.filter(i => i !== q.index));
    }
  }), /*#__PURE__*/React.createElement("span", null, q.question, /*#__PURE__*/React.createElement("small", null, q.sourceMatched ? 'Source matched · model reviewed; still check the premise' : 'Premise needs source review'))))), !pack.questions.length ? /*#__PURE__*/React.createElement("p", null, "No usable diligence questions were saved in this report.") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("label", null, "Meeting date", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: when,
    onInput: e => {
      setWhen(e.currentTarget.value);
      setSaved(null);
    },
    onChange: e => {
      setWhen(e.target.value);
      setSaved(null);
    }
  })), /*#__PURE__*/React.createElement("label", null, "Meeting type", /*#__PURE__*/React.createElement("select", {
    value: kind,
    onChange: e => {
      setKind(e.target.value);
      setSaved(null);
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: "management"
  }, "Management"), /*#__PURE__*/React.createElement("option", {
    value: "expert"
  }, "Expert"), /*#__PURE__*/React.createElement("option", {
    value: "conference"
  }, "Conference"), /*#__PURE__*/React.createElement("option", {
    value: "other"
  }, "Other")))), /*#__PURE__*/React.createElement("button", {
    disabled: !indices.length || !when,
    onClick: save
  }, busy ? 'Saving…' : `Save ${indices.length} selected questions to Meeting Prep`))), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), saved && /*#__PURE__*/React.createElement("button", {
    onClick: () => onMeeting?.(saved.meetingId)
  }, "Open saved meeting"));
}