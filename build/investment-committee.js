import * as React from 'react';
export function CommitteeFindings({
  run
}) {
  var claims = new Map((run.state?.sections || []).flatMap(s => s.claims.map(c => [c.id, {
    ...c,
    role: s.title
  }])));
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Committee challenges"
  }, /*#__PURE__*/React.createElement("h4", null, "Challenge round \xB7 draft responses"), /*#__PURE__*/React.createElement("p", null, "Initial assessments are preserved below. These are five roles of the same configured model, not five independent people or a vote. An addressed challenge means the lead proposed a response; it is not evidence verification or investor approval."), (run.state?.challenges || []).map(c => {
    var response = run.state?.responses?.find(r => r.challengeId === c.id);
    return /*#__PURE__*/React.createElement("article", {
      key: c.id
    }, /*#__PURE__*/React.createElement("h4", null, c.question), c.claimIds.map(id => /*#__PURE__*/React.createElement("p", {
      key: id
    }, /*#__PURE__*/React.createElement("strong", null, claims.get(id)?.role || id, ":"), " ", claims.get(id)?.statement || 'Assessment unavailable')), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, response?.status || 'Response pending')), response && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, response.reason), /*#__PURE__*/React.createElement("p", null, "Next test: ", response.nextTest), /*#__PURE__*/React.createElement("p", null, "Unaccepted proposal: ", response.proposedChange)));
  }), run.state?.completed?.includes('challenges') && !run.state.challenges.length && /*#__PURE__*/React.createElement("p", null, "No challenges returned in this round. This is not proof of consensus or complete coverage."), /*#__PURE__*/React.createElement("p", null, "Review proposed changes in Current thesis or Evidence & proposals. This committee cannot approve or edit your case."));
}