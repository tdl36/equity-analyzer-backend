import * as React from 'react';
export function ProcessingSelectionReview({
  summary: s,
  purpose,
  onRemove
}) {
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Evidence review before save",
    style: {
      margin: '16px 0',
      padding: 12,
      border: '1px solid var(--border-color,#d8dedd)',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Review the combined evidence pack"), /*#__PURE__*/React.createElement("p", null, s.total, " selected originals \xB7 ", s.collection, " collected \xB7 ", s.supplemental, " supplemental history. These are your current choices, not a saved or approved proposal."), s.metadataPending && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Source metadata is not fully loaded or its latest read failed. Review loaded selections below; load or retry the source lists before saving."), s.missing > 0 && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, s.missing, " selected originals have no loaded metadata. Load the remaining library entries or explicitly remove those selections; they have not been silently dropped."), s.unavailable > 0 && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, s.unavailable, " selected originals are missing, ambiguous or restricted. Resolve their availability or remove them before saving."), /*#__PURE__*/React.createElement("p", null, s.dates.known ? `Selected publication dates: ${s.dates.earliest} – ${s.dates.latest}.` : 'No selected publication dates recorded.', " ", s.dates.unknown, " dates unknown. This range does not establish continuous coverage; gaps between documents have not been assessed."), s.dates.windowKnown ? /*#__PURE__*/React.createElement("p", null, s.dates.before, " dated before the collection window \xB7 ", s.dates.within, " within it \xB7 ", s.dates.after, " after it.") : /*#__PURE__*/React.createElement("p", null, "The collection date window is unavailable; date comparisons cannot be made."), /*#__PURE__*/React.createElement("p", null, purpose === 'event_note' ? 'Historical selections provide context for the identified event. This note does not update the thesis or establish thesis freshness.' : 'The collection window limits discovery, not the longitudinal investment case. Keep relevant foundational and contrary evidence; dates alone do not establish sufficient history.'), s.duplicates.length > 0 && /*#__PURE__*/React.createElement("div", {
    role: "alert"
  }, /*#__PURE__*/React.createElement("strong", null, "Possible duplicate copies in this selection"), /*#__PURE__*/React.createElement("p", null, "Matching recorded hashes appear below. Inspect these sources and keep the intended copy. Metadata is not a current-byte verification; saving checks actual originals and refuses duplicate bytes."), /*#__PURE__*/React.createElement("ul", null, s.duplicates.map(group => /*#__PURE__*/React.createElement("li", {
    key: group[0].key
  }, group.map(r => r.filename).join(' · '))))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, s.hashUnknown, " selections have no recorded hash for this preview. Different or absent recorded hashes do not prove unique content. Exact-byte checks, size limits and thesis-register references are evaluated when saving; no research credits are used.")), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Selected originals \xB7 ", s.total), /*#__PURE__*/React.createElement("ul", null, s.rows.map(r => /*#__PURE__*/React.createElement("li", {
    key: r.key,
    style: {
      padding: '8px 0'
    }
  }, r.filename, /*#__PURE__*/React.createElement("small", {
    style: {
      display: 'block'
    }
  }, r.role === 'collection' ? 'Collected' : 'Supplemental history', " \xB7 Published: ", r.date || 'unknown', " \xB7 ", r.metadata ? (r.metadata.typeRecorded ? 'Recorded type: ' : 'Type suggested from filename: ') + (r.metadata.type || 'unclassified') : 'Metadata not loaded'), /*#__PURE__*/React.createElement("button", {
    "aria-label": 'Remove ' + r.role + ' source ' + r.filename,
    onClick: () => onRemove(r)
  }, "Remove from selection")))), !s.total && /*#__PURE__*/React.createElement("p", null, "Select originals above to review the pack.")), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Not selected: ", s.unselectedCollection === null ? 'collection inventory not loaded' : s.unselectedCollection + ' collection receipts', "; ", s.unselectedLibrary, " loaded library entries. ", s.libraryComplete ? 'All pages of this supplemental inventory have been loaded.' : 'The supplemental library has not been fully loaded; additional unselected sources may exist.', " Unselected includes restricted entries and is not a relevance judgment. Collection and library inventories do not certify every relevant source is available. Accepted-thesis incorporation remains unassessed.")));
}