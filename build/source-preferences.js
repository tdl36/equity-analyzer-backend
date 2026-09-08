import * as React from 'react';
export function SourcePreferences({
  api,
  onChange
}) {
  var [saved, setSaved] = React.useState(null),
    [draft, setDraft] = React.useState(null),
    [error, setError] = React.useState(''),
    [message, setMessage] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [lists, setLists] = React.useState([]);
  var call = async (path, body, method = 'PUT') => {
    var r = await fetch(api + path, {
      ...(body ? {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      } : {}),
      signal: AbortSignal.timeout(20000)
    });
    var d = await r.json();
    if (!r.ok) throw Error(d.error || 'Source preferences unavailable');
    return d;
  };
  var reload = async () => {
    try {
      var d = await call('/api/research/source-preferences');
      setSaved(d);
      setDraft(d.policy);
      onChange(d.policy);
      setError('');
    } catch (e) {
      setError(e.message);
    }
  };
  React.useEffect(() => {
    var active = true;
    call('/api/research/source-preferences').then(d => {
      if (active) {
        setSaved(d);
        setDraft(d.policy);
        onChange(d.policy);
      }
    }).catch(e => {
      if (active) setError(e.message);
    });
    var poll = () => call('/api/research/source-shortlists').then(d => {
      if (active) setLists(d.shortlists);
    }).catch(e => {
      if (active) setError(e.message);
    });
    poll();
    var timer = setInterval(poll, 15000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [api]);
  var edit = p => {
    setDraft(p);
    onChange(p);
    setMessage('Assignment override · save to reuse across devices.');
  };
  var save = async () => {
    setBusy(true);
    try {
      var d = await call('/api/research/source-preferences', {
        revision: saved.revision,
        policy: draft
      });
      setSaved(d);
      setDraft(d.policy);
      onChange(d.policy);
      setMessage('Defaults saved across devices. Existing assignments keep their original policy.');
      setError('');
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  var choose = async (list, url, choice) => {
    setBusy(true);
    try {
      await call(`/api/research/commands/${list.commandId}/source-shortlist`, {
        revision: list.revision,
        choices: {
          [url]: choice
        }
      });
      setLists((await call('/api/research/source-shortlists')).shortlists);
      setMessage('Choice saved. Resume the paused request in Collection and recovery controls when your selection is complete.');
      setError('');
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel source-preferences"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "RESEARCH STANDARDS"), /*#__PURE__*/React.createElement("h2", null, "Your sources. Your judgment."), /*#__PURE__*/React.createElement("p", null, "Primary disclosures anchor facts. Broker research adds interpretation. Set your preferences once, or adjust them for this assignment."), draft && /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy
  }, /*#__PURE__*/React.createElement("label", null, "Who chooses broker research?", /*#__PURE__*/React.createElement("select", {
    value: draft.mode,
    onChange: e => edit({
      ...draft,
      mode: e.target.value
    })
  }, /*#__PURE__*/React.createElement("option", {
    value: "auto"
  }, "Charlie decides \xB7 respect my preferences"), /*#__PURE__*/React.createElement("option", {
    value: "preferred"
  }, "Preferred sources only \xB7 ask before exceptions"), /*#__PURE__*/React.createElement("option", {
    value: "review"
  }, "Let me select \xB7 review the document shortlist"))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Broker preferences & company overrides (", draft.rules.length, ")"), /*#__PURE__*/React.createElement("p", null, "Use the publisher name exactly as shown in AlphaSense. Add alternate spellings as separate rules. A company/task rule overrides the general rule. Reference-only sources stay outside AI analysis."), draft.rules.map((r, i) => /*#__PURE__*/React.createElement("div", {
    className: "source-rule",
    key: i
  }, /*#__PURE__*/React.createElement("label", null, "Broker / source", /*#__PURE__*/React.createElement("input", {
    maxLength: 120,
    value: r.name,
    onChange: e => edit({
      ...draft,
      rules: draft.rules.map((x, j) => j === i ? {
        ...x,
        name: e.target.value
      } : x)
    })
  })), /*#__PURE__*/React.createElement("label", null, "Preference", /*#__PURE__*/React.createElement("select", {
    value: r.disposition,
    onChange: e => edit({
      ...draft,
      rules: draft.rules.map((x, j) => j === i ? {
        ...x,
        disposition: e.target.value
      } : x)
    })
  }, ['preferred', 'standard', 'excluded', 'reference_only'].map(x => /*#__PURE__*/React.createElement("option", {
    key: x,
    value: x
  }, x.replace('_', ' '))))), /*#__PURE__*/React.createElement("label", null, "Ticker \xB7 optional", /*#__PURE__*/React.createElement("input", {
    maxLength: 20,
    placeholder: "All companies",
    value: r.ticker,
    onChange: e => edit({
      ...draft,
      rules: draft.rules.map((x, j) => j === i ? {
        ...x,
        ticker: e.target.value.toUpperCase()
      } : x)
    })
  })), /*#__PURE__*/React.createElement("label", null, "Task", /*#__PURE__*/React.createElement("select", {
    value: r.task,
    onChange: e => edit({
      ...draft,
      rules: draft.rules.map((x, j) => j === i ? {
        ...x,
        task: e.target.value
      } : x)
    })
  }, ['', 'meeting', 'filing', 'earnings', 'event'].map(x => /*#__PURE__*/React.createElement("option", {
    key: x,
    value: x
  }, x || 'All tasks')))), /*#__PURE__*/React.createElement("button", {
    type: "button",
    "aria-label": `Remove source rule ${i + 1}`,
    onClick: () => edit({
      ...draft,
      rules: draft.rules.filter((_, j) => j !== i)
    })
  }, "Remove"))), /*#__PURE__*/React.createElement("button", {
    type: "button",
    disabled: draft.rules.length >= 80,
    onClick: () => edit({
      ...draft,
      rules: [...draft.rules, {
        name: '',
        disposition: 'preferred',
        ticker: '',
        task: ''
      }]
    })
  }, "Add a source preference")), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Charlie should prioritize expertise, original analysis, relevance, freshness and credible disagreement\u2014not brand prestige or the number of similar notes. Provider usage restrictions always apply. These defaults apply to new Command Charlie assignments; existing scheduled ticker policies are separate."), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: save
  }, "Save as my defaults"), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: reload
  }, "Reload saved preferences")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: reload
  }, "Reload")), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), lists.map(list => /*#__PURE__*/React.createElement("details", {
    key: list.commandId
  }, /*#__PURE__*/React.createElement("summary", null, list.ticker, " \xB7 source shortlist \xB7 ", list.candidates.filter(x => x.decision === 'pending').length, " awaiting selection"), /*#__PURE__*/React.createElement("p", null, "Decisions apply to these specific documents. Exclusions and provider restrictions still override an include choice."), list.candidates.map(c => /*#__PURE__*/React.createElement("article", {
    className: "amendment-card",
    key: c.url
  }, /*#__PURE__*/React.createElement("strong", null, c.title), /*#__PURE__*/React.createElement("p", null, c.publisher, " \xB7 ", c.decision), /*#__PURE__*/React.createElement("p", null, c.reason), /*#__PURE__*/React.createElement("a", {
    href: c.url,
    target: "_blank",
    rel: "noreferrer"
  }, "Inspect in AlphaSense \u2197"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || c.decision === 'include',
    onClick: () => choose(list, c.url, 'include')
  }, "Include"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || c.decision === 'exclude',
    onClick: () => choose(list, c.url, 'exclude')
  }, "Exclude"))))));
}