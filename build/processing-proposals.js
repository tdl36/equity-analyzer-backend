import { processingSelection } from './processing-selection-model.mjs';
import { ProcessingSelectionReview } from './processing-selection-review';
import { OriginalDownload } from './original-download';
import { CollectedSourcePicker } from './collected-source-picker';
import { processingProposal, processingProposalPage, mergeProcessingProposals } from './processing-proposal-status.mjs';
import { emptyDraft, emptyPurposeDraft, draftKey, readDraftWorkspace, writeDraftWorkspace, readSubmission } from './processing-proposal-draft.mjs';
import { ProcessingCoverage } from './processing-coverage';
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
    key: api + ':' + job.id,
    api: api,
    job: job
  }));
}
function ProposalReview({
  api,
  job
}) {
  var localKey = draftKey(api, job.id),
    [restored] = React.useState(() => {
      try {
        return readDraftWorkspace(sessionStorage, localKey);
      } catch {
        return {
          value: null,
          error: 'Local draft storage is unavailable. Saved proposals are unaffected.'
        };
      }
    });
  var initial = restored.value || emptyDraft();
  var purposeDrafts = React.useRef(restored.drafts || {});
  var [submissionIssue, setSubmissionIssue] = React.useState('');
  var [draftIssue, setDraftIssue] = React.useState(restored.error),
    [restoredNotice, setRestoredNotice] = React.useState(!!restored.value);
  var [fullNote, setFullNote] = React.useState(initial.fullNote),
    [allowPdfText, setAllowPdfText] = React.useState(initial.allowPdfText),
    [meetingDate, setMeetingDate] = React.useState(initial.meetingDate),
    [meetingType, setMeetingType] = React.useState(initial.meetingType);
  var [collectionInventory, setCollectionInventory] = React.useState({
      rows: [],
      loaded: false
    }),
    [libraryInventory, setLibraryInventory] = React.useState({
      rows: [],
      loaded: false
    });
  var [collectionReady, setCollectionReady] = React.useState(false);
  var [supplements, setSupplements] = React.useState(initial.supplements);
  var [selected, setSelected] = React.useState(initial.selected),
    [purpose, setPurpose] = React.useState(initial.purpose),
    [title, setTitle] = React.useState(initial.title),
    [eventDate, setEventDate] = React.useState(initial.eventDate),
    [outputs, setOutputs] = React.useState(initial.outputs),
    [instruction, setInstruction] = React.useState(initial.instruction);
  var [rows, setRows] = React.useState([]),
    [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState(''),
    [readIssue, setReadIssue] = React.useState(''),
    [loaded, setLoaded] = React.useState(false),
    [pending, setPending] = React.useState(null);
  var [nextCursor, setNextCursor] = React.useState(null),
    [historyIssue, setHistoryIssue] = React.useState('');
  var readEpoch = React.useRef(0);
  var lock = React.useRef(false),
    live = React.useRef(true);
  var selectionSummary = processingSelection({
    selected,
    supplements,
    collection: collectionInventory,
    library: libraryInventory,
    since: job.input.since,
    until: job.input.until
  });
  var authoring = {
    version: 1,
    selected,
    supplements,
    purpose,
    title,
    eventDate,
    outputs,
    instruction,
    fullNote,
    allowPdfText,
    meetingDate,
    meetingType
  };
  React.useLayoutEffect(() => {
    if (draftIssue || pending) return;
    try {
      writeDraftWorkspace(sessionStorage, localKey, authoring, purposeDrafts.current);
      purposeDrafts.current[purpose] = authoring;
    } catch {
      setDraftIssue('Unsaved choices cannot be kept on this device. Save a proposal before leaving.');
    }
  }, [localKey, JSON.stringify(authoring), draftIssue, pending]);
  var applyDraft = d => {
    setSelected(d.selected);
    setSupplements(d.supplements);
    setPurpose(d.purpose);
    setTitle(d.title);
    setEventDate(d.eventDate);
    setOutputs(d.outputs);
    setInstruction(d.instruction);
    setFullNote(d.fullNote);
    setAllowPdfText(d.allowPdfText);
    setMeetingDate(d.meetingDate);
    setMeetingType(d.meetingType);
  };
  var changePurpose = next => {
    purposeDrafts.current[purpose] = authoring;
    applyDraft(purposeDrafts.current[next] || emptyPurposeDraft(next));
    setRestoredNotice(false);
  };
  var resetForm = () => {
    var d = emptyPurposeDraft(purpose);
    purposeDrafts.current[purpose] = d;
    applyDraft(d);
    setRestoredNotice(false);
    try {
      writeDraftWorkspace(sessionStorage, localKey, d, purposeDrafts.current);
      setDraftIssue('');
    } catch {
      setDraftIssue('Current draft could not be cleared in browser recovery. Keep this screen open.');
    }
  };
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
    var epoch = ++readEpoch.current;
    var current = () => live.current && epoch === readEpoch.current;
    try {
      var data = await call(path),
        page = processingProposalPage(data, job);
      var proposals = page.rows;
      if (!current()) return;
      if (current()) {
        setRows(proposals);
        setNextCursor(page.nextCursor);
        setLoaded(true);
      }
      // Browser recovery must not hide verified server records.
      var recovered;
      try {
        recovered = readSubmission(sessionStorage, key);
      } catch {
        recovered = readSubmission(null, key);
      }
      var issue = recovered.error;
      if (recovered.value) {
        if (live.current) setPending(recovered.value);
        if (!proposals.some(p => p.id === recovered.value.requestId)) {
          var exact = processingProposalPage(await call(path + '?requestId=' + encodeURIComponent(recovered.value.requestId)), job, {
            requestId: recovered.value.requestId
          });
          proposals = mergeProcessingProposals(proposals, exact.rows);
        }
        if (!current()) return;
        if (proposals.some(p => p.id === recovered.value.requestId)) {
          try {
            sessionStorage.removeItem(key);
            if (live.current) {
              setPending(null);
              resetForm();
            }
          } catch {
            issue = 'The saved proposal was found, but its local recovery record could not be cleared. Restore browser storage and refresh before submitting again.';
          }
        } else if (live.current) setPending(recovered.value);
      } else if (!issue && live.current) setPending(null);
      if (current()) {
        setSubmissionIssue(issue);
        setRows(proposals);
        setNextCursor(page.nextCursor);
        setHistoryIssue('');
        setLoaded(true);
        setReadIssue('');
      }
    } catch (e) {
      if (!current()) return;
      setReadIssue(e.message);
      throw e;
    }
  };
  var older = async () => {
    if (lock.current || !nextCursor) return;
    lock.current = true;
    setBusy(true);
    setHistoryIssue('');
    var cursor = nextCursor,
      epoch = readEpoch.current;
    try {
      var page = processingProposalPage(await call(path + '?after=' + encodeURIComponent(cursor)), job, {
        after: cursor
      });
      if (live.current && epoch === readEpoch.current) {
        setRows(old => mergeProcessingProposals(old, page.rows));
        setNextCursor(page.nextCursor);
      }
    } catch (e) {
      if (live.current && epoch === readEpoch.current) setHistoryIssue(e.message);
    } finally {
      lock.current = false;
      if (live.current) setBusy(false);
    }
  };
  React.useEffect(() => {
    live.current = true;
    refresh().catch(e => live.current && setError(e.message));
    return () => {
      live.current = false;
      readEpoch.current++;
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
    if (submissionIssue) throw Error(submissionIssue);
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
        horizon: '12–24 months',
        ...(fullNote ? {
          labFullNote: {
            allowPdfText
          }
        } : {}),
        ...(outputs.includes('meeting_prep') ? {
          meeting: {
            date: meetingDate,
            type: meetingType
          }
        } : {})
      }
    };
    try {
      sessionStorage.setItem(key, JSON.stringify(body));
    } catch {
      setSubmissionIssue('Submission was not sent because browser recovery could not be saved. Restore browser storage and refresh before submitting.');
      throw Error('No proposal was sent. Browser recovery storage is unavailable.');
    }
    setPending(body);
    try {
      var receipt = processingProposal(await call(path, body), job);
      if (receipt.id !== body.requestId) throw Error('Proposal receipt mismatch. Refresh saved proposals.');
      sessionStorage.removeItem(key);
      setPending(null);
      resetForm();
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
  }, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Collected library \u2192 selected evidence \u2192 your processing approval")), /*#__PURE__*/React.createElement("p", null, "Collection searched ", job.input.since, " through ", job.input.until, ". This is collection coverage, not proof that the accepted thesis incorporates these documents. Saving a proposal spends no research credits."), readIssue && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, readIssue), error && error !== readIssue && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), restoredNotice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Restored unsaved choices in this browser tab. Review them; originals and baseline are checked again when you save. No processing was requested."), draftIssue && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, draftIssue), submissionIssue && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, submissionIssue), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Unsaved choices are kept in this browser tab while storage is available. Saving a proposal creates the durable server record."), " ", /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!pending,
    onClick: resetForm
  }, "Discard current purpose draft")), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(async () => {})
  }, "Refresh saved proposals"), pending && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "A proposal submission is unresolved. Refresh to find its receipt, or retry the same saved submission below. Its choices are frozen."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || !loaded || !!pending || !!readIssue
  }, /*#__PURE__*/React.createElement("legend", null, "Select evidence and purpose"), /*#__PURE__*/React.createElement("label", null, "Processing purpose", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Processing purpose",
    value: purpose,
    onChange: e => changePurpose(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "thesis_review"
  }, "Thesis review \xB7 longitudinal"), /*#__PURE__*/React.createElement("option", {
    value: "event_note"
  }, "Event note \xB7 specific catalyst"))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Each purpose keeps its own unfinished sources, outputs and instructions in this browser tab. Switching restores that purpose\u2019s draft; it never copies or submits the other one.")), /*#__PURE__*/React.createElement("p", null, purpose === 'thesis_review' ? 'Review the full investment case over time. Add relevant foundational and historical originals from the wider library below. The collection window does not limit the investment case; missing evidence remains a gap.' : 'Debrief an identified event with contextual history. This does not change the thesis or mark it current. Thesis implications need a separate proposal.'), purpose === 'event_note' && /*#__PURE__*/React.createElement("div", {
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
  }))), /*#__PURE__*/React.createElement("p", null, selected.length, " of ", job.result.sources.length, " collected originals selected, plus ", supplements.length, " supplemental originals \xB7 none selected automatically. Select at least one collected original to link this proposal to the collection. Up to 15 MB per original and 60 MB per processing pack; collection itself can be larger."), /*#__PURE__*/React.createElement(CollectedSourcePicker, {
    api: api,
    call: call,
    job: job,
    selected: selected,
    onChange: setSelected,
    purpose: purpose,
    remaining: 1000 - supplements.length,
    onReady: setCollectionReady,
    onInventory: setCollectionInventory
  }), /*#__PURE__*/React.createElement(ProcessingSourceLibrary, {
    api: api,
    ticker: job.ticker,
    call: call,
    path: path.replace('processing-proposals', 'processing-library'),
    selected: supplements,
    onChange: setSupplements,
    purpose: purpose,
    remaining: 1000 - selected.length,
    onInventory: setLibraryInventory
  }), /*#__PURE__*/React.createElement(ProcessingSelectionReview, {
    summary: selectionSummary,
    purpose: purpose,
    onRemove: r => r.role === 'collection' ? setSelected(selected.filter(n => n !== r.selection)) : setSupplements(supplements.filter(id => id !== r.selection))
  }), /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, assignmentOutputs.filter(([id]) => purpose !== 'event_note' || id === 'summary_note' || id === 'meeting_prep').map(([id, label]) => /*#__PURE__*/React.createElement("label", {
    className: "sa-check",
    key: id
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: outputs.includes(id),
    onChange: e => setOutputs(p => e.target.checked ? [...p, id] : p.filter(n => n !== id))
  }), label))), outputs.includes('meeting_prep') && /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("label", null, "Meeting date", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Processing meeting date",
    type: "date",
    value: meetingDate,
    onChange: e => setMeetingDate(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Meeting type", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Processing meeting type",
    value: meetingType,
    onChange: e => setMeetingType(e.target.value)
  }, ['management', 'expert', 'conference', 'other'].map(t => /*#__PURE__*/React.createElement("option", {
    key: t,
    value: t
  }, t)))), /*#__PURE__*/React.createElement("p", null, "Meeting questions use the shared research report; no additional generation stage.")), /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: fullNote,
    onChange: e => setFullNote(e.target.checked)
  }), "Full versioned Lab note \xB7 additional research credits"), fullNote && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", null, "Saving checks source readability and batch limits without model calls, then freezes the model, instructions and source representations. The full note uses up to eight additional model calls, capped at 32,000 output tokens per call. Large evidence packs are processed in bounded stages; omissions remain visible."), /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: allowPdfText,
    onChange: e => setAllowPdfText(e.target.checked)
  }), "Allow extracted PDF text when needed \xB7 charts and layout may be omitted")), /*#__PURE__*/React.createElement("label", null, "Focus and evidence gaps", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    maxLength: 1800,
    value: instruction,
    onChange: e => setInstruction(e.target.value)
  }))), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !loaded || !!readIssue || !!submissionIssue || !pending && (!collectionReady || selectionSummary.metadataPending || selectionSummary.missing > 0 || selectionSummary.unavailable > 0 || !selected.length || !outputs.length || outputs.includes('meeting_prep') && !meetingDate || purpose === 'event_note' && (!title.trim() || !eventDate)),
    onClick: save
  }, pending ? 'Retry saved proposal submission' : 'Save proposal · no generation'), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Approval starts the existing shared research workflow: up to twelve model calls within the configured budget. Large evidence packs may be excerpted with omissions recorded; selection does not guarantee exhaustive review. No exact credit estimate is available. Finished research still requires investor review.")), !loaded ? /*#__PURE__*/React.createElement("p", null, readIssue ? 'Saved proposals unavailable. Refresh to check their status.' : 'Loading saved proposals…') : !rows.length ? /*#__PURE__*/React.createElement("p", null, "No saved processing proposals for this collection.") : rows.map(p => /*#__PURE__*/React.createElement("article", {
    key: p.id,
    style: {
      borderTop: '1px solid var(--border-color,#d8dedd)',
      padding: '14px 0'
    }
  }, /*#__PURE__*/React.createElement("strong", null, p.plan.researchPurpose.kind === 'event_note' ? p.plan.researchPurpose.event.title : 'Longitudinal thesis review'), " \xB7 ", p.status, /*#__PURE__*/React.createElement("p", null, p.plan.researchPurpose.kind === 'event_note' ? `Event ${p.plan.researchPurpose.event.date} · ` : '', p.sources.length, " verified originals at save \xB7 ", p.plan.outputs.map(id => assignmentOutputs.find(x => x[0] === id)?.[1] || id).join(', ')), p.plan.meeting && /*#__PURE__*/React.createElement("p", null, "Meeting preparation \xB7 ", p.plan.meeting.date, " \xB7 ", p.plan.meeting.type), p.fullNote && /*#__PURE__*/React.createElement("p", null, "Full versioned note \xB7 frozen model ", p.fullNote.modelKey, " \xB7 up to ", p.fullNote.maxCalls, " additional calls, ", p.fullNote.maxTokens.toLocaleString(), " output tokens per call \xB7 PDF text fallback ", p.fullNote.allowPdfText ? 'allowed' : 'off', ". Draft requires investor review."), p.fullNote?.sourcePreparation && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", null, "Full-note source check: ", p.fullNote.sourcePreparation.sourcePack.length, " originals planned across ", p.fullNote.sourcePreparation.batchCount, " batches. Rechecked before dispatch; provider context validation still happens at execution."), p.fullNote.sourcePreparation.warnings.length > 0 && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source representation limitations \xB7 ", p.fullNote.sourcePreparation.warnings.length), /*#__PURE__*/React.createElement("ul", null, p.fullNote.sourcePreparation.warnings.map(w => /*#__PURE__*/React.createElement("li", {
    key: w.reference
  }, p.sources.find(s => s.reference === w.reference)?.filename || w.reference, ": ", w.issue))))), p.plan.instruction && /*#__PURE__*/React.createElement("p", null, p.plan.instruction), /*#__PURE__*/React.createElement(ProcessingCoverage, {
    coverage: p.coverage
  }), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Frozen sources and scope"), /*#__PURE__*/React.createElement("p", null, "Revision ", p.revision, ". Collection ", p.plan.since, "\u2013", p.plan.until, ". Sources and baseline will be checked again before dispatch."), p.fullNote && /*#__PURE__*/React.createElement("p", null, "Full-note configuration: ", p.fullNote.configDigest, /*#__PURE__*/React.createElement("br", null), "Note instructions: ", p.fullNote.promptSha256), /*#__PURE__*/React.createElement("ul", null, p.sources.map(s => /*#__PURE__*/React.createElement("li", {
    key: s.documentId
  }, s.filename, " ", /*#__PURE__*/React.createElement(OriginalDownload, {
    url: api + '/api/pipeline-lab/original/' + encodeURIComponent(p.ticker) + '/thesis_' + s.documentId,
    filename: s.filename,
    sha256: s.sha256
  }), " \xB7 ", (s.bytes / 1000000).toFixed(2), " MB", p.sourceContext?.[s.documentId] && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, p.sourceContext[s.documentId].role === 'supplemental' ? 'Supplemental history; not collected by this request' : 'From collection receipt', " \xB7 Published: ", p.sourceContext[s.documentId].publishedDate || 'unknown', " \xB7 ", p.sourceContext[s.documentId].referencedBySavedThesis ? 'Exact bytes referenced by saved thesis; incorporation still needs review' : 'No exact-hash reference found in saved thesis register; incorporation unknown')))))), p.status === 'pending' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy || !!readIssue || !loaded,
    onClick: () => act(async () => {
      await call('/api/research/processing-proposals/' + p.id + '/approve', {
        revision: p.revision,
        digest: p.digest
      });
    })
  }, "Approve processing \xB7 uses research credits"), " ", /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!readIssue || !loaded,
    onClick: () => act(async () => {
      await call('/api/research/processing-proposals/' + p.id + '/cancel', {
        revision: p.revision,
        digest: p.digest
      });
    })
  }, "Cancel proposal")), p.assignmentId && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    href: routeHash('stockanalysis', p.ticker, 'stock-analysis', 'assignment:' + p.assignmentId)
  }, "Open saved processing job and outputs")))), loaded && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, rows.length, " saved proposals loaded", nextCursor ? ' · older proposals remain' : '', ". Refresh returns to the latest page.")), historyIssue && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "Older proposal history unavailable: ", historyIssue, " Loaded proposals are retained."), nextCursor && /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!readIssue,
    onClick: older
  }, historyIssue ? 'Retry older proposals' : 'Load older proposals'));
}