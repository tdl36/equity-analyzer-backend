import * as React from 'react';
export default function NoteProvenance({
  value
}) {
  if (!value) return /*#__PURE__*/React.createElement("p", null, "This note has no structured frozen research-purpose or source-date record. Its scope is not inferred from its title.");
  var event = value.purpose.kind === 'event_note' ? value.purpose.event : null;
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Saved research scope",
    style: {
      background: '#fff',
      color: '#000',
      fontFamily: 'Calibri, sans-serif',
      padding: 16,
      margin: '16px 0',
      border: '1px solid #ddd',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("h4", {
    style: {
      marginTop: 0
    }
  }, event ? 'Event note / debrief' : 'Longitudinal thesis review'), event ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, event.title), " \xB7 ", event.date), /*#__PURE__*/React.createElement("p", null, "Event-focused analysis with historical context. This note does not update the thesis or establish thesis freshness. Thesis implications require a separate proposal.")) : /*#__PURE__*/React.createElement("p", null, "Discovery dates limit new-source collection, not analytical history. This note does not apply thesis changes."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Frozen source context \xB7 ", value.sourceEvidence?.sources.length ?? 'not recorded'), /*#__PURE__*/React.createElement("p", null, "Publication dates are not financial periods. Thesis-register references do not prove incorporation or current validity. Missing references do not establish that evidence is new."), value.sourceEvidence ? /*#__PURE__*/React.createElement("ul", {
    style: {
      paddingLeft: 20
    }
  }, value.sourceEvidence.sources.map(s => /*#__PURE__*/React.createElement("li", {
    key: s.id,
    style: {
      marginBottom: 12
    }
  }, /*#__PURE__*/React.createElement("strong", null, s.filename), /*#__PURE__*/React.createElement("br", null), s.role === 'collection' ? 'Collected evidence' : 'Supplemental history', " \xB7 Publication: ", s.publishedDate || 'Unknown', /*#__PURE__*/React.createElement("br", null), "Publisher: ", s.publisher || 'Unknown', " \xB7 Type: ", s.sourceKind || 'Unknown', /*#__PURE__*/React.createElement("br", null), s.referencedBySavedThesis === true ? 'Referenced by saved thesis register' : s.referencedBySavedThesis === false ? 'No saved thesis-register reference' : 'Thesis-register reference unknown'))) : /*#__PURE__*/React.createElement("p", null, "Source dates and roles were not recorded for this note.")));
}