import * as React from 'react';
import { originalBlob, originalFilename } from './original-download.mjs';
export function OriginalDownload({
  url,
  filename,
  sha256
}) {
  var [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState('');
  var lock = React.useRef(false);
  var download = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      // Use the application's authenticated fetch; credentials never appear in a URL.
      var blob = await originalBlob(await fetch(url, {
        signal: AbortSignal.timeout(30000)
      }), sha256);
      var link = document.createElement('a'),
        objectUrl = URL.createObjectURL(blob);
      try {
        link.href = objectUrl;
        link.download = originalFilename(filename);
        document.body.appendChild(link);
        link.click();
      } finally {
        link.remove();
        setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
      }
    } catch (e) {
      setError(e.message || 'Original download could not be verified.');
    } finally {
      lock.current = false;
      setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("button", {
    type: "button",
    disabled: busy,
    onClick: download
  }, busy ? 'Checking original…' : 'Download original'), error && /*#__PURE__*/React.createElement("small", {
    role: "alert",
    style: {
      display: 'block'
    }
  }, error));
}