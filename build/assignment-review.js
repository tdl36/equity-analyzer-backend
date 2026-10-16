import * as React from 'react';
import { PackageEditor } from './package-editor';
import { reviewDocuments, assignmentLabNote, verifiedReviewPage } from './assignment-review-model.mjs';
import { AssignmentMeetingOutput } from './assignment-meeting-output';
import { HistoricalReview } from './historical-review';
import { downloadText } from './stock-analysis-model.mjs';
export function AssignmentReview({
  job,
  api
}) {
  var note = assignmentLabNote(job),
    docs = reviewDocuments(job),
    signature = JSON.stringify(docs),
    [selected, setSelected] = React.useState(docs[0]?.id),
    [originalKey, setOriginalKey] = React.useState(null),
    [retry, setRetry] = React.useState(0),
    [loadedPage, setPage] = React.useState({}),
    frame = React.useRef(null);
  var doc = docs.find(d => d.id === selected) || docs[0];
  var selectionKey = JSON.stringify([api, job.id, job.ticker, job.result?.reportId, signature, doc?.id]),
    original = !!doc?.originalUrl && originalKey === selectionKey;
  var pageKey = JSON.stringify([selectionKey, original]),
    page = loadedPage.key === pageKey ? loadedPage : {};
  React.useEffect(() => {
    var live = true;
    var controller = new AbortController();
    setPage({
      key: pageKey,
      loading: !!doc?.url
    });
    if (doc?.url) (async () => {
      try {
        var r = await fetch(api + (original ? doc.originalUrl : doc.url), {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(60000)])
        });
        if (!r.ok) {
          var failure = await r.json().catch(() => null);
          throw Error(failure?.error || 'This saved document could not be opened. Retry loading; no research will be started.');
        }
        var saved = doc.json ? verifiedReviewPage(await r.json(), doc, job) : {
          html: await r.text()
        };
        if (!saved.html) throw Error('The saved preview is unavailable.');
        if (live) setPage({
          key: pageKey,
          ...saved
        });
      } catch (e) {
        if (live) setPage({
          key: pageKey,
          error: e.message
        });
      }
    })();
    return () => {
      live = false;
      controller.abort();
    };
  }, [pageKey, retry]);
  if (!docs.length && !note) return null;
  return /*#__PURE__*/React.createElement("section", {
    className: "assignment-review",
    "aria-label": "Review saved outputs"
  }, /*#__PURE__*/React.createElement("style", null, styles), /*#__PURE__*/React.createElement("header", null, /*#__PURE__*/React.createElement("small", null, job.ticker, " / SAVED RESEARCH"), /*#__PURE__*/React.createElement("h2", null, !docs.length ? 'Research delivery' : job.status === 'complete' ? 'Ready for your review' : 'Available drafts'), /*#__PURE__*/React.createElement("p", null, job.input.since, " \u2013 ", job.input.until, " \xB7 ", job.input.horizon || 'Saved research horizon'), /*#__PURE__*/React.createElement("p", {
    className: "review-caution"
  }, job.result.reviewPackage ? 'Recovered drafts from the same frozen originals. Original automated outputs are preserved in history below.' : 'Review the evidence and limitations before using these drafts.', " Opening a preview does not approve or change your thesis.")), job.result.reviewPackage && /*#__PURE__*/React.createElement(PackageEditor, {
    key: job.result.reviewPackage.requestId,
    job: job,
    api: api
  }), note && !note.saved && /*#__PURE__*/React.createElement("div", {
    className: "review-evidence",
    role: "status"
  }, /*#__PURE__*/React.createElement("h3", null, "Full Lab note \xB7 ", note.status), /*#__PURE__*/React.createElement("p", null, note.status === 'Needs attention' ? 'Inspect the saved note progress before any retry. Other completed outputs remain available.' : 'A saved full-note draft has not been confirmed. Collection or parent completion alone does not establish delivery.'), note.href ? /*#__PURE__*/React.createElement("a", {
    className: "review-link",
    href: note.href
  }, "Open full-note progress and recovery") : /*#__PURE__*/React.createElement("p", null, "No note identity is available yet. Check Sources, progress & history below for the assignment\u2019s next step.")), docs.length > 0 && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    role: "tablist",
    "aria-label": "Saved deliverables"
  }, docs.map((d, i) => /*#__PURE__*/React.createElement("button", {
    key: d.id,
    id: 'review-tab-' + job.id + '-' + d.id,
    role: "tab",
    "aria-selected": d.id === doc?.id,
    "aria-controls": 'review-panel-' + job.id,
    tabIndex: d.id === doc?.id ? 0 : -1,
    onClick: () => setSelected(d.id),
    onKeyDown: e => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
        e.preventDefault();
        var n = e.key === 'Home' ? 0 : e.key === 'End' ? docs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + docs.length) % docs.length;
        setSelected(docs[n].id);
        e.currentTarget.parentElement.children[n].focus();
      }
    }
  }, d.label))), /*#__PURE__*/React.createElement("div", {
    role: "tabpanel",
    id: 'review-panel-' + job.id,
    "aria-labelledby": 'review-tab-' + job.id + '-' + doc?.id
  }, doc?.meetingOutput ? /*#__PURE__*/React.createElement(AssignmentMeetingOutput, {
    api: api,
    job: job,
    doc: doc
  }) : doc?.href ? /*#__PURE__*/React.createElement("div", {
    className: "review-decision"
  }, /*#__PURE__*/React.createElement("h3", null, doc.label), /*#__PURE__*/React.createElement("p", null, doc.description), /*#__PURE__*/React.createElement("a", {
    className: "review-link",
    href: doc.href
  }, doc.actionLabel)) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "review-toolbar"
  }, /*#__PURE__*/React.createElement("strong", null, doc?.label), /*#__PURE__*/React.createElement("span", null, page.status === 'accepted' ? 'Investor-accepted edition' : page.status === 'pending' ? 'Pending edition — review required' : original ? 'Original saved output' : doc?.originalUrl ? 'Rendered from saved report' : 'Saved preview', " \xB7 no generation"), /*#__PURE__*/React.createElement("button", {
    disabled: page.loading,
    onClick: () => setRetry(x => x + 1)
  }, original ? 'Reload original output' : 'Refresh saved version'), doc?.originalUrl && /*#__PURE__*/React.createElement("button", {
    disabled: page.loading,
    onClick: () => setOriginalKey(original ? null : selectionKey)
  }, original ? 'Render from saved report' : 'Open original saved output'), /*#__PURE__*/React.createElement("button", {
    disabled: !page.html,
    onClick: () => downloadText(page.html, `${job.ticker}-${doc.id}-${original ? 'original' : 'review'}.html`)
  }, "Download"), /*#__PURE__*/React.createElement("button", {
    disabled: !page.html,
    onClick: () => frame.current?.contentWindow?.print()
  }, "Print")), original && /*#__PURE__*/React.createElement("p", {
    className: "review-caution",
    role: "status"
  }, "Original automated output, preserved as delivered. Later layout changes, financial supplements and investor editions are not included. This is not an investor acceptance or a fresh analysis."), page.loading && /*#__PURE__*/React.createElement("p", {
    role: "status",
    className: "review-message"
  }, "Opening your saved ", doc.label.toLowerCase(), "\u2026"), page.error && /*#__PURE__*/React.createElement("div", {
    role: "alert",
    className: "review-message"
  }, page.error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setRetry(x => x + 1)
  }, "Retry loading")), page.html && /*#__PURE__*/React.createElement("iframe", {
    ref: frame,
    title: `${job.ticker} ${doc.label} review`,
    sandbox: "allow-same-origin allow-modals",
    srcDoc: page.html
  }))), doc?.id === 'visual' && job.result?.reviewPackage?.editionId && /*#__PURE__*/React.createElement("div", {
    className: "review-evidence"
  }, /*#__PURE__*/React.createElement(HistoricalReview, {
    api: api,
    assignmentId: job.id,
    editionId: job.result.reviewPackage.editionId
  }))));
}
var styles = `.review-evidence{padding:18px 24px;border-top:1px solid #ddd;overflow-wrap:anywhere}.review-evidence summary{cursor:pointer;font-weight:700}.review-evidence p{line-height:1.5}.assignment-focused{padding:0!important}.assignment-focused .sa-prepare{padding:0!important;background:transparent!important;border:0!important}.assignment-focused .sa-prepare>article{padding:0!important;border:0!important}.assignment-focused .sa-prepare>p{margin:8px 0 16px}.assignment-focused .assignment-review{margin-top:0}.assignment-focused .assignment-review [role=tabpanel]{padding:0}.assignment-focused .assignment-review header{border:0}.assignment-focused .assignment-history{padding:16px!important}.assignment-review{font-family:Calibri,sans-serif;color:#111;background:#fff;border:1px solid #d9d9d2;border-radius:10px;overflow:hidden;margin:8px 0 28px}.assignment-review header{padding:28px 30px 16px}.assignment-review header small{letter-spacing:.13em;font-size:12px}.assignment-review h2{font-family:Calibri,sans-serif!important;color:#111!important;font-size:30px!important;margin:8px 0!important}.assignment-review [role=tabpanel]>.review-caution{margin:12px 24px;padding:12px 16px;background:#f5f4ee;line-height:1.5;font-size:14px}.assignment-review header p{margin:8px 0;line-height:1.5}.assignment-review .review-caution{max-width:85ch;font-size:14px;color:#111;background:#f5f4ee;padding:12px 16px}.assignment-review [role=tablist]{display:flex;flex-wrap:wrap;gap:4px;padding:0 24px;border-bottom:1px solid #ccc}.assignment-review [role=tab]{border:0!important;border-radius:0!important;border-bottom:3px solid transparent!important;background:transparent!important;padding:14px 16px!important;margin:0!important;color:#111!important;font-size:17px!important}.assignment-review [role=tab][aria-selected=true]{border-bottom-color:#111!important;font-weight:700;background:#f6f5ef!important}.assignment-review button:focus-visible,.assignment-review a:focus-visible{outline:2px solid #111;outline-offset:3px}.assignment-review .review-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:10px;padding:12px 24px;background:#fafaf7}.review-toolbar span{flex:1;font-size:13px}.assignment-review .review-toolbar button{margin:0!important}.assignment-review iframe{display:block;width:100%;height:76vh;min-height:560px;border:0;border-top:1px solid #ddd;background:white}.review-message,.review-decision{padding:30px;min-height:180px}.review-link{display:inline-block;padding:12px 18px;border:1px solid #111;border-radius:4px;color:#111!important;font-weight:bold}.assignment-history{margin:18px 0;padding:16px;border:1px solid #ddd;border-radius:8px}.assignment-history>summary{font-weight:bold;cursor:pointer}.assignment-review~p{overflow-wrap:anywhere}@media(max-width:650px){.assignment-review header{padding:20px 16px 12px}.assignment-review h2{font-size:26px!important}.assignment-review [role=tablist]{padding:0 8px}.assignment-review [role=tab]{font-size:15px!important;padding:12px 10px!important}.assignment-review .review-toolbar{padding:12px}.review-toolbar span{flex-basis:100%;order:4}.assignment-review iframe{min-height:480px;height:72vh}.review-message,.review-decision{padding:20px}}`;