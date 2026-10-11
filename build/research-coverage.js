import * as React from 'react';
import { researchCoverage, coverageDocument } from './research-coverage-model.mjs';
import { downloadText } from './stock-analysis-model.mjs';
export function ResearchCoverage({
  run,
  onSection
}) {
  var c = React.useMemo(() => researchCoverage(run), [run]);
  var [filter, setFilter] = React.useState('review');
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Research coverage and review"
  }, /*#__PURE__*/React.createElement("h4", null, "What this research can support"), /*#__PURE__*/React.createElement("p", null, c.matched, " matched fields \xB7 ", c.review, " need source review \xB7 ", c.unavailable, " unavailable items"), /*#__PURE__*/React.createElement("p", null, "Counts include labels and periods. A matched excerpt and a model review do not establish factual accuracy, complete market coverage or investor approval."), c.baselineStale && /*#__PURE__*/React.createElement("p", {
    className: "sa-warning"
  }, "Your investment case changed after this report started. Compare the current case with the frozen baseline before using this report."), c.noEvidence.length > 0 && /*#__PURE__*/React.createElement("p", {
    className: "sa-warning"
  }, /*#__PURE__*/React.createElement("strong", null, "No matched fields:"), " ", c.noEvidence.join(' · '), ". Open these sections to inspect the gaps before relying on the conclusions."), /*#__PURE__*/React.createElement("p", null, c.sources.length, " frozen originals \xB7 ", c.limitedSources, " with recorded extraction/selection limits \xB7 ", c.unknownCoverage, " without coverage metadata"), /*#__PURE__*/React.createElement("div", {
    className: "sa-actions"
  }, /*#__PURE__*/React.createElement("button", {
    "aria-pressed": filter === 'review',
    onClick: () => setFilter('review')
  }, "Needs source review"), /*#__PURE__*/React.createElement("button", {
    "aria-pressed": filter === 'all',
    onClick: () => setFilter('all')
  }, "All sections"), /*#__PURE__*/React.createElement("button", {
    onClick: () => downloadText(coverageDocument(run), `${run.ticker}-review-checklist-${run.id}.html`)
  }, "Export review checklist")), /*#__PURE__*/React.createElement("div", {
    style: {
      overflowX: 'auto'
    }
  }, /*#__PURE__*/React.createElement("table", {
    style: {
      width: '100%',
      borderCollapse: 'collapse'
    }
  }, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Section"), /*#__PURE__*/React.createElement("th", null, "Matched"), /*#__PURE__*/React.createElement("th", null, "Review"), /*#__PURE__*/React.createElement("th", null, "Unavailable"))), /*#__PURE__*/React.createElement("tbody", null, c.rows.filter(r => filter === 'all' || r.review || !r.matched || r.unavailable).map(r => /*#__PURE__*/React.createElement("tr", {
    key: r.id
  }, /*#__PURE__*/React.createElement("th", null, /*#__PURE__*/React.createElement("button", {
    onClick: () => onSection(r.id)
  }, "Open ", r.title)), /*#__PURE__*/React.createElement("td", null, r.matched), /*#__PURE__*/React.createElement("td", null, r.review), /*#__PURE__*/React.createElement("td", null, r.unavailable)))))), filter === 'review' && !c.rows.some(r => r.review || !r.matched || r.unavailable) && /*#__PURE__*/React.createElement("p", null, "No field-level review gaps recorded. Investor review is still required."), c.rows.filter(r => r.review).map(r => /*#__PURE__*/React.createElement("details", {
    key: r.id
  }, /*#__PURE__*/React.createElement("summary", null, r.title, " \xB7 ", r.review, " fields to check"), r.fields.filter(f => f.status === 'review').map(f => /*#__PURE__*/React.createElement("article", {
    key: f.path
  }, /*#__PURE__*/React.createElement("small", null, f.path, " \xB7 ", f.basis.replaceAll('_', ' ')), /*#__PURE__*/React.createElement("p", null, f.value), /*#__PURE__*/React.createElement("p", {
    className: "sa-warning"
  }, f.reason))), /*#__PURE__*/React.createElement("button", {
    onClick: () => onSection(r.id)
  }, "Open section and source passages"))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source coverage limits \xB7 ", c.sources.length, " originals"), c.sources.map(s => /*#__PURE__*/React.createElement("article", {
    key: s.id
  }, /*#__PURE__*/React.createElement("strong", null, s.name), /*#__PURE__*/React.createElement("p", null, s.used ? 'Referenced by at least one matched field.' : 'No matched report field references this source.'), /*#__PURE__*/React.createElement("p", null, s.limitation || (!s.coverageKnown ? 'Extraction coverage metadata unavailable; full-document review is not established.' : 'No recorded selection omission; this is not proof of complete document review.')), s.omitted != null && /*#__PURE__*/React.createElement("p", null, s.omitted.toLocaleString(), " extracted characters omitted from model input."), s.extraction.map((v, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, v))))));
}