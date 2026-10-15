import React from 'react';
import { checkedLabImport, checkedLabImports, checkedLabImportRequest, reconcileLabImport } from './lab-import-receipts.mjs';
export default function LabICloudImport({
  api,
  ticker,
  task,
  revision,
  selected,
  disabled,
  onImported,
  read
}) {
  var [jobs, setJobs] = React.useState([]),
    [ready, setReady] = React.useState(false),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [pending, setPending] = React.useState(false);
  var alive = React.useRef(true),
    lock = React.useRef(false),
    seen = React.useRef(null),
    callback = React.useRef(onImported);
  callback.current = onImported;
  var key = `charlie-lab-icloud-v1:${api}:${ticker}`;
  var refresh = async () => {
    try {
      var d = await read(api, `/api/pipeline-lab/icloud-imports/${ticker}`);
      var rows = checkedLabImports(d, ticker);
      if (!alive.current) return;
      var completed = rows.filter(j => j.status === 'complete').map(j => j.requestId);
      var changed = seen.current && completed.some(id => !seen.current.has(id));
      seen.current = new Set(completed);
      setJobs(rows);
      setReady(true);
      setError('');
      if (changed) callback.current();
    } catch (e) {
      if (alive.current) {
        setError(e.message);
        setReady(false);
      }
    }
  };
  React.useEffect(() => {
    alive.current = true;
    try {
      setPending(!!sessionStorage.getItem(key));
    } catch {
      setError('Retry storage unavailable; imports cannot start safely.');
    }
    refresh();
    var timer = setInterval(refresh, 10000);
    return () => {
      alive.current = false;
      clearInterval(timer);
    };
  }, [api, ticker]);
  var run = async (action, job) => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var result;
      if (action === 'start') {
        var prior = sessionStorage.getItem(key);
        var body = checkedLabImportRequest(prior ? JSON.parse(prior) : {
          requestId: crypto.randomUUID(),
          task,
          revision
        });
        if (!prior) sessionStorage.setItem(key, JSON.stringify(body));
        if (sessionStorage.getItem(key) !== JSON.stringify(body)) throw Error('Could not preserve request identity. Nothing was submitted.');
        setPending(true);
        result = prior ? await reconcileLabImport(read, api, ticker, body) : null;
        if (!result) result = await read(api, `/api/pipeline-lab/icloud-imports/${ticker}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(body)
        });
        checkedLabImport(result, ticker, body);
        sessionStorage.removeItem(key);
        setPending(false);
      } else {
        result = await read(api, `/api/pipeline-lab/icloud-import/${job.requestId}/${action}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: '{}'
        });
        checkedLabImport(result, ticker, {
          requestId: job.requestId,
          ...job.input
        });
      }
      if (result.status === 'complete') callback.current();
      await refresh();
    } catch (e) {
      if (action === 'start' && [400, 409].includes(e.status)) {
        try {
          var old = JSON.parse(sessionStorage.getItem(key));
          await read(api, `/api/pipeline-lab/icloud-import/${old.requestId}`);
        } catch (check) {
          if (check.status === 404) {
            sessionStorage.removeItem(key);
            if (alive.current) setPending(false);
          }
        }
      }
      if (alive.current) setError(e.message + ' Existing originals remain preserved.');
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var active = jobs.find(j => ['queued', 'frozen', 'attention'].includes(j.status));
  return /*#__PURE__*/React.createElement("section", {
    style: {
      marginTop: 20
    },
    "aria-label": "iCloud original imports"
  }, /*#__PURE__*/React.createElement("h3", null, "Import selected iCloud originals"), /*#__PURE__*/React.createElement("p", null, "Copies the exact selected folders and files into Charlie, then updates only this task\u2019s selection. Originals stay in iCloud. Requires the Mac online, its agent running and files downloaded. No research starts."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), (selected.some(r => r.startsWith('icloud:')) || pending) && /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy || !ready || !pending && !!active,
    onClick: () => run('start')
  }, pending ? 'Reconcile same iCloud request' : 'Import selected iCloud originals'), active && /*#__PURE__*/React.createElement("div", {
    role: "status"
  }, /*#__PURE__*/React.createElement("p", null, "Import ", active.status, " \xB7 ", active.input.task, " sources \xB7 saved revision ", active.input.revision), active.issue && /*#__PURE__*/React.createElement("p", null, active.issue), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, active.status === 'queued' ? 'Waiting for the Mac agent. This does not confirm the files are available.' : active.status === 'frozen' ? 'Original hashes recorded; waiting for verified cloud receipts.' : 'Resolve the issue, then retry the same frozen originals or cancel and save a new selection.')), active.status === 'attention' && /*#__PURE__*/React.createElement("button", {
    disabled: busy || disabled,
    onClick: () => run('retry', active)
  }, "Retry this import"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => run('cancel', active)
  }, "Cancel this import")), jobs.filter(j => j.status === 'complete').slice(0, 1).map(j => /*#__PURE__*/React.createElement("p", {
    key: j.requestId
  }, "Last completed import: ", j.receipts.length, " original(s) verified \xB7 ", j.input.task, " selection updated to revision ", j.completedRevision, ". No research was started.")), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: refresh
  }, "Refresh import status"));
}