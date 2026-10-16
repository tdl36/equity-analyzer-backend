import * as React from 'react';
import { checkedProcessingReadiness } from './processing-readiness-model.mjs';
export function ProcessingReadiness({
  proposal,
  call,
  path,
  disabled,
  onApprove
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false);
  var live = React.useRef(true),
    lock = React.useRef(false);
  React.useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  var check = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var result = checkedProcessingReadiness(await call(path + '/' + encodeURIComponent(proposal.id) + '/readiness'), proposal);
      if (live.current) setData(result);
    } catch (e) {
      if (live.current) setError(e.message);
    } finally {
      lock.current = false;
      if (live.current) setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Processing readiness",
    style: {
      padding: '12px 0',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("p", null, "Check the saved sources and baseline before approving. This read checks original bytes, permissions, thesis/research baseline and local dispatch prerequisites. It starts no generation and spends no research credits."), /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy,
    onClick: check
  }, busy ? 'Checking saved proposal…' : data ? 'Recheck processing readiness' : 'Check processing readiness · no generation'), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " Any retained check is out of date. Recheck before approving."), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, data.ready ? 'No blocker found at this check.' : 'Processing is blocked.', " Checked ", new Date(data.checkedAt).toLocaleString(), "."), data.blockers.length > 0 && /*#__PURE__*/React.createElement("ul", null, data.blockers.map((b, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, b))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "This is not a reservation, provider availability guarantee or approval. Approval checks the current state again; finished research still requires investor review."))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: disabled || busy || !!error || !data?.ready,
    onClick: () => {
      setData(null);
      onApprove();
    }
  }, "Approve processing \xB7 uses research credits"));
}