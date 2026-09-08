import * as React from 'react';
export function SourcePreferences({
  api,
  onChange
}) {
  var [mapTicker, setMapTicker] = React.useState(''),
    [mapSector, setMapSector] = React.useState(''),
    [previewTicker, setPreviewTicker] = React.useState(''),
    [previewBroker, setPreviewBroker] = React.useState(''),
    [previewTask, setPreviewTask] = React.useState('meeting'),
    [resolved, setResolved] = React.useState(null);
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
    setResolved(null);
    setMessage('');
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
    setResolved(null);
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
  }, "Let me select \xB7 review the document shortlist"))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Broker preferences & company overrides (", draft.rules.length, ")"), /*#__PURE__*/React.createElement("p", null, "Use the publisher name exactly as shown in AlphaSense. Add alternate spellings as separate rules. Stock rules override subsector rules; within a scope, named analysts override broker-wide rules. Task-specific rules break ties. Author identity must be observed on the report, never inferred from a firm or previous coverage. Reference-only sources stay outside AI analysis."), draft.rules.map((r, i) => /*#__PURE__*/React.createElement("div", {
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
        ticker: e.target.value.toUpperCase(),
        subsector: ''
      } : x)
    })
  })), /*#__PURE__*/React.createElement("label", null, "Subsector \xB7 optional", /*#__PURE__*/React.createElement("input", {
    disabled: !!r.ticker,
    maxLength: 100,
    value: r.subsector || '',
    placeholder: "All subsectors",
    onChange: e => edit({
      ...draft,
      rules: draft.rules.map((x, j) => j === i ? {
        ...x,
        subsector: e.target.value
      } : x)
    })
  })), /*#__PURE__*/React.createElement("label", null, "Named analyst \xB7 optional", /*#__PURE__*/React.createElement("input", {
    maxLength: 120,
    value: r.analyst || '',
    placeholder: "Any analyst at this broker",
    onChange: e => edit({
      ...draft,
      rules: draft.rules.map((x, j) => j === i ? {
        ...x,
        analyst: e.target.value
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
  }, "Add a source preference")), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Company subsectors & effective preferences"), /*#__PURE__*/React.createElement("p", null, "Use your own coverage classifications. Subsector rules apply only to companies mapped here. No classification is inferred."), /*#__PURE__*/React.createElement("div", {
    className: "source-rule"
  }, /*#__PURE__*/React.createElement("label", null, "Ticker", /*#__PURE__*/React.createElement("input", {
    value: mapTicker,
    maxLength: 20,
    onChange: e => setMapTicker(e.target.value.toUpperCase()),
    placeholder: "UNH"
  })), /*#__PURE__*/React.createElement("label", null, "Subsector", /*#__PURE__*/React.createElement("input", {
    value: mapSector,
    maxLength: 100,
    onChange: e => setMapSector(e.target.value),
    placeholder: "Healthcare services"
  })), /*#__PURE__*/React.createElement("button", {
    type: "button",
    disabled: !mapTicker.trim() || !mapSector.trim(),
    onClick: () => {
      edit({
        ...draft,
        subsectors: {
          ...draft.subsectors,
          [mapTicker.trim()]: mapSector.trim()
        }
      });
      setMapTicker('');
      setMapSector('');
    }
  }, "Assign subsector")), Object.entries(draft.subsectors || {}).map(([tk, sector]) => /*#__PURE__*/React.createElement("p", {
    key: tk
  }, /*#__PURE__*/React.createElement("strong", null, tk), " \xB7 ", sector, " ", /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => {
      var next = {
        ...draft.subsectors
      };
      delete next[tk];
      edit({
        ...draft,
        subsectors: next
      });
    }
  }, "Remove ", tk, " classification"))), /*#__PURE__*/React.createElement("h3", null, "Which rule will Charlie use?"), /*#__PURE__*/React.createElement("div", {
    className: "source-rule"
  }, /*#__PURE__*/React.createElement("label", null, "Check ticker", /*#__PURE__*/React.createElement("input", {
    value: previewTicker,
    onChange: e => {
      setPreviewTicker(e.target.value.toUpperCase());
      setResolved(null);
    }
  })), /*#__PURE__*/React.createElement("label", null, "Check broker", /*#__PURE__*/React.createElement("input", {
    value: previewBroker,
    onChange: e => {
      setPreviewBroker(e.target.value);
      setResolved(null);
    }
  })), /*#__PURE__*/React.createElement("label", null, "Check task", /*#__PURE__*/React.createElement("select", {
    value: previewTask,
    onChange: e => {
      setPreviewTask(e.target.value);
      setResolved(null);
    }
  }, ['meeting', 'filing', 'earnings', 'event'].map(x => /*#__PURE__*/React.createElement("option", {
    key: x
  }, x)))), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: async () => {
      try {
        setError('');
        setResolved(await call('/api/research/source-preferences/resolve', {
          policy: draft,
          ticker: previewTicker,
          publisher: previewBroker,
          task: previewTask
        }, 'POST'));
      } catch (e) {
        setError(e.message);
      }
    }
  }, "Explain preference")), resolved && /*#__PURE__*/React.createElement("div", {
    role: "status"
  }, /*#__PURE__*/React.createElement("strong", null, resolved.decision, " \xB7 ", resolved.disposition), /*#__PURE__*/React.createElement("p", null, resolved.reason, " Subsector: ", resolved.subsector || 'not assigned', "."), resolved.rule && /*#__PURE__*/React.createElement("p", null, "Applied rule: ", resolved.rule.name, " \xB7 ", resolved.rule.ticker || resolved.rule.subsector || 'all companies', " \xB7 ", resolved.rule.analyst || 'all analysts', " \xB7 ", resolved.rule.task || 'all tasks', "."), /*#__PURE__*/React.createElement("p", null, "Preview assumes the report author is not yet verified. Named-analyst rules are checked against observed authorship during collection."))), /*#__PURE__*/React.createElement("p", {
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
  }, /*#__PURE__*/React.createElement("strong", null, c.title), /*#__PURE__*/React.createElement("p", null, c.publisher, " \xB7 ", c.analyst || 'Author not verified', " \xB7 ", c.decision), c.authorEvidence && /*#__PURE__*/React.createElement("p", null, "Observed authorship: ", c.authorEvidence), /*#__PURE__*/React.createElement("p", null, c.reason), /*#__PURE__*/React.createElement("a", {
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