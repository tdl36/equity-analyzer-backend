import * as React from 'react';

// Consent belongs to this saved assignment state, never to the surrounding list.
export function AssignmentResume({
  job,
  disabled,
  onResume
}) {
  var [acknowledged, setAcknowledged] = React.useState(null);
  var revision = typeof job.resumeRevision === 'string' && /^[a-f0-9]{64}$/.test(job.resumeRevision) ? job.resumeRevision : null;
  var scope = revision ? job.id + ':' + revision : null;
  var checked = !!scope && acknowledged === scope;
  React.useEffect(() => {
    if (disabled) setAcknowledged(null);
  }, [disabled]);
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Assignment resume controls"
  }, !revision && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Retry controls need a current saved-state receipt. Wait for assignment status to refresh."), /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    disabled: disabled || !revision,
    checked: checked,
    onChange: e => setAcknowledged(e.target.checked ? scope : null)
  }), "If a paid call is uncertain, I checked usage and accept that retry may charge again."), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "This acknowledgment applies only to this assignment\u2019s current saved state and is cleared after each attempt.")), /*#__PURE__*/React.createElement("button", {
    disabled: disabled || !revision,
    onClick: () => {
      setAcknowledged(null);
      onResume({
        acknowledgeRetry: checked,
        expectedRevision: revision
      });
    }
  }, "Resume assignment"));
}