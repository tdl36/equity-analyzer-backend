import { ProcessingSourceLibrary } from './processing-source-library';
import * as React from 'react';
import { routeHash } from './workspace-model.mjs';
import { assignmentOutputs } from './assignment-defaults.mjs';
export function ProcessingProposals({
  api,
  job
}) {
  var [open, setOpen] = React.useState(false);
  return /*#__PURE__*/React.createElement("details", {
    onToggle: e => {
      if (e.currentTarget.open) setOpen(true);
    }
  }, /*#__PURE__*/React.createElement("summary", null, "Choose sources & prepare processing approval"), open && /*#__PURE__*/React.createElement(ProposalReview, {
    api: api,
    job: job
  }));
}
function ProposalReview({
  api,
  job
}) {
  var [supplements, setSupplements] = React.useState([]);
  var [selected, setSelected] = React.useState([]),
    [purpose, setPurpose] = React.useState('thesis_review'),
    [title, setTitle] = React.useState(''),
    [eventDate, setEventDate] = React.useState(''),
    [outputs, setOutputs] = React.useState(['thesis']),
    [instruction, setInstruction] = React.useState('');
  var [rows, setRows] = React.useState([]),
    [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState(''),
    [loaded, setLoaded] = React.useState(false),
    [pending, setPending] = React.useState(null);
  var lock = React.useRef(false),
    live = React.useRef(true);
  var key = 'charlie-processing-proposal:' + api + ':' + job.id;
  var path = '/api/research/assignments/' + job.id + '/processing-proposals';
  var call = async (url, body) => {
    var response = await fetch(api + url, {
      ...(body ? {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      } : {}),
      signal: AbortSignal.timeout(30000)
    });
    var data;
    try {
      data = await response.json();
    } catch {
      throw Error('Response unavailable. Refresh saved proposals before retrying.');
    }
    if (!response.ok) {
      var e = Error(data.error || 'Processing proposals unavailable');
      e.status = response.status;
      throw e;
    }
    return data;
  };
  var refresh = async () => {
    var data = await call(path);
    if (!Array.isArray(data.proposals)) throw Error('Proposal list unavailable. No processing was requested.');
    if (live.current) {
      setRows(data.proposals);
      setLoaded(true);
    }
    var raw = sessionStorage.getItem(key);
    if (raw) {
      var saved = JSON.parse(raw);
      if (data.proposals.some(p => p.id === saved.requestId)) {
        sessionStorage.removeItem(key);
        if (live.current) setPending(null);
      }
    }
  };
  React.useEffect(() => {
    live.current = true;
    try {
      var raw = sessionStorage.getItem(key);
      if (raw) setPending(JSON.parse(raw));
    } catch {
      setError('Saved submission could not be read. Inspect saved proposals before creating another.');
      return;
    }
    refresh().catch(e => live.current && setError(e.message));
    return () => {
      live.current = false;
    };
  }, [key]);
  var act = async fn => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      await fn();
      await refresh();
    } catch (e) {
      if (live.current) setError(e.message);
    } finally {
      lock.current = false;
      if (live.current) setBusy(false);
    }
  };
  var save = () => act(async () => {
    var body = pending || {
      requestId: crypto.randomUUID(),
      filenames: [...selected].sort(),
      supplementalIds: [...supplements].sort(),
      plan: {
        researchPurpose: purpose === 'event_note' ? {
          kind: purpose,
          event: {
            title: title.trim(),
            date: eventDate
          }
        } : {
          kind: purpose
        },
        outputs: [...outputs].sort(),
        instruction,
        horizon: '12–24 months'
      }
    };
    sessionStorage.setItem(key, JSON.stringify(body));
    setPending(body);
    try {
      var receipt = await call(path, body);
      if (receipt.id !== body.requestId) throw Error('Proposal receipt mismatch. Refresh saved proposals.');
      sessionStorage.removeItem(key);
      setPending(null);
    } catch (e) {
      if (e.status === 409) {
        sessionStorage.removeItem(key);
        setPending(null);
      }
      throw e;
    }
  });
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Processing approval",
    style: {
      padding: '12px 0',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Collected library \u2192 selected evidence \u2192 your processing approval")), /*#__PURE__*/React.createElement("p", null, "Collection searched ", job.input.since, " through ", job.input.until, ". This is collection coverage, not proof that the accepted thesis incorporates these documents. Saving a proposal spends no research credits."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(async () => {})
  }, "Refresh saved proposals"), pending && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "A proposal submission is unresolved. Refresh to find its receipt, or retry the same saved submission below. Its choices are frozen."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || !loaded || !!pending
  }, /*#__PURE__*/React.createElement("legend", null, "Select evidence and purpose"), /*#__PURE__*/React.createElement("label", null, "Processing purpose", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Processing purpose",
    value: purpose,
    onChange: e => {
      setPurpose(e.target.value);
      setOutputs(e.target.value === 'event_note' ? ['summary_note'] : ['thesis']);
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: "thesis_review"
  }, "Thesis review \xB7 longitudinal"), /*#__PURE__*/React.createElement("option", {
    value: "event_note"
  }, "Event note \xB7 specific catalyst"))), /*#__PURE__*/React.createElement("p", null, purpose === 'thesis_review' ? 'Review the full investment case over time. Add relevant foundational and historical originals from the wider library below. The collection window does not limit the investment case; missing evidence remains a gap.' : 'Debrief an identified event with contextual history. This does not change the thesis or mark it current. Thesis implications need a separate proposal.'), purpose === 'event_note' && /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("label", null, "Event title", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Processing event title",
    maxLength: 200,
    value: title,
    onChange: e => setTitle(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Event date", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Processing event date",
    type: "date",
    value: eventDate,
    onChange: e => setEventDate(e.target.value)
  }))), /*#__PURE__*/React.createElement("p", null, selected.length, " of ", job.result.sources.length, " collected originals selected, plus ", supplements.length, " supplemental originals \xB7 none selected automatically. Select at least one collected original to link this proposal to the collection. Up to 15 MB per original and 60 MB per processing pack; collection itself can be larger."), /*#__PURE__*/React.createElement("div", {
    style: {
      maxHeight: 280,
      overflow: 'auto',
      border: '1px solid var(--border-color,#d8dedd)',
      padding: 12
    }
  }, job.result.sources.map(s => /*#__PURE__*/React.createElement("label", {
    className: "sa-check",
    key: s.filename,
    style: {
      display: 'flex',
      alignItems: 'start',
      gap: 8,
      padding: '6px 0'
    }
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: selected.includes(s.filename),
    disabled: !selected.includes(s.filename) && selected.length + supplements.length >= 1000,
    onChange: e => setSelected(p => e.target.checked ? [...p, s.filename] : p.filter(n => n !== s.filename))
  }), /*#__PURE__*/React.createElement("span", null, s.filename)))), /*#__PURE__*/React.createElement(ProcessingSourceLibrary, {
    call: call,
    path: path.replace('processing-proposals', 'processing-library'),
    selected: supplements,
    onChange: setSupplements,
    purpose: purpose,
    remaining: 1000 - selected.length
  }), /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, assignmentOutputs.filter(([id]) => id !== 'meeting_prep' && (purpose !== 'event_note' || id === 'summary_note')).map(([id, label]) => /*#__PURE__*/React.createElement("label", {
    className: "sa-check",
    key: id
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: outputs.includes(id),
    onChange: e => setOutputs(p => e.target.checked ? [...p, id] : p.filter(n => n !== id))
  }), label))), /*#__PURE__*/React.createElement("label", null, "Focus and evidence gaps", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    maxLength: 1800,
    value: instruction,
    onChange: e => setInstruction(e.target.value)
  }))), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !loaded || !pending && (!selected.length || !outputs.length || purpose === 'event_note' && (!title.trim() || !eventDate)),
    onClick: save
  }, pending ? 'Retry saved proposal submission' : 'Save proposal · no generation'), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Approval starts the existing shared research workflow: up to twelve model calls within the configured budget. Large evidence packs may be excerpted with omissions recorded; selection does not guarantee exhaustive review. No exact credit estimate is available. Finished research still requires investor review.")), !loaded ? /*#__PURE__*/React.createElement("p", null, "Loading saved proposals\u2026") : !rows.length ? /*#__PURE__*/React.createElement("p", null, "No saved processing proposals for this collection.") : rows.map(p => /*#__PURE__*/React.createElement("article", {
    key: p.id,
    style: {
      borderTop: '1px solid var(--border-color,#d8dedd)',
      padding: '14px 0'
    }
  }, /*#__PURE__*/React.createElement("strong", null, p.plan.researchPurpose.kind === 'event_note' ? p.plan.researchPurpose.event.title : 'Longitudinal thesis review'), " \xB7 ", p.status, /*#__PURE__*/React.createElement("p", null, p.plan.researchPurpose.kind === 'event_note' ? `Event ${p.plan.researchPurpose.event.date} · ` : '', p.sources.length, " verified originals at save \xB7 ", p.plan.outputs.map(id => assignmentOutputs.find(x => x[0] === id)?.[1] || id).join(', ')), p.plan.instruction && /*#__PURE__*/React.createElement("p", null, p.plan.instruction), p.sourceContext && Object.keys(p.sourceContext).length > 0 && /*#__PURE__*/React.createElement("p", null, Object.values(p.sourceContext).filter(s => s.role === 'collection').length, " from the collection \xB7 ", Object.values(p.sourceContext).filter(s => s.role === 'supplemental').length, " supplemental history. Collection coverage and thesis evidence coverage are different."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Frozen sources and scope"), /*#__PURE__*/React.createElement("p", null, "Revision ", p.revision, ". Collection ", p.plan.since, "\u2013", p.plan.until, ". Sources and baseline will be checked again before dispatch."), /*#__PURE__*/React.createElement("ul", null, p.sources.map(s => /*#__PURE__*/React.createElement("li", {
    key: s.documentId
  }, /*#__PURE__*/React.createElement("a", {
    href: api + '/api/pipeline-lab/original/' + encodeURIComponent(p.ticker) + '/thesis_' + s.documentId,
    target: "_blank",
    rel: "noreferrer"
  }, s.filename), " \xB7 ", (s.bytes / 1000000).toFixed(2), " MB", p.sourceContext?.[s.documentId] && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, p.sourceContext[s.documentId].role === 'supplemental' ? 'Supplemental history; not collected by this request' : 'From collection receipt', " \xB7 Published: ", p.sourceContext[s.documentId].publishedDate || 'unknown', " \xB7 ", p.sourceContext[s.documentId].referencedBySavedThesis ? 'Exact bytes referenced by saved thesis; incorporation still needs review' : 'No exact-hash reference found in saved thesis register; incorporation unknown')))))), p.status === 'pending' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy,
    onClick: () => act(async () => {
      await call('/api/research/processing-proposals/' + p.id + '/approve', {
        revision: p.revision,
        digest: p.digest
      });
    })
  }, "Approve processing \xB7 uses research credits"), " ", /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(async () => {
      await call('/api/research/processing-proposals/' + p.id + '/cancel', {
        revision: p.revision,
        digest: p.digest
      });
    })
  }, "Cancel proposal")), p.assignmentId && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    href: routeHash('stockanalysis', p.ticker, 'stock-analysis', 'assignment:' + p.assignmentId)
  }, "Open saved processing job and outputs")))));
}