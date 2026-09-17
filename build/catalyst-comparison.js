import * as React from 'react';
import { emailContent, shareableViews, qaSpeakerVisibility } from './catalyst-sharing.mjs';

// Isolated catalyst viewer; does not use or change Summary generation or storage.
export function CatalystComparison({
  value,
  jobId,
  ticker,
  api,
  renderHtml
}) {
  var [view, setView] = React.useState('pm');
  var [showSpeakers, setShowSpeakers] = React.useState(true);
  var [message, setMessage] = React.useState('');
  var [busy, setBusy] = React.useState(false);
  var lock = React.useRef(false);
  var [emailScope, setEmailScope] = React.useState(null);
  var [recipient, setRecipient] = React.useState('');
  React.useEffect(() => {
    setMessage('');
    setView('pm');
    setEmailScope(null);
  }, [jobId, value?.sourceHash]);
  if (!value) return null;
  var labels = value.workflowMode === 'event' ? {
    pm: 'Event note',
    quick: 'Evidence & review'
  } : value.editorialVersion ? {
    pm: 'PM takeaway',
    comprehensive: 'Detailed note',
    ...(value.qa ? {
      qa: 'Q&A'
    } : {}),
    quick: 'Evidence & review'
  } : {
    pm: 'Brief',
    quick: 'Source record',
    summary: 'Interpretation',
    comprehensive: 'Full note'
  };
  var sections = {};
  for (var match of (value.markdown || '').matchAll(/<section data-version="(pm|quick|summary|comprehensive|qa)">([\s\S]*?)<\/section>/g)) sections[match[1]] = match[2];
  if (sections.qa) sections.qa = qaSpeakerVisibility(sections.qa, showSpeakers);
  var selected = sections[view] || value.markdown || '';
  var shareable = shareableViews(value, sections);
  var prepareEmail = scope => {
    var creds = JSON.parse(localStorage.getItem('emailCredentials') || '{}');
    setRecipient(creds.email || '');
    setEmailScope(scope);
    setMessage('');
  };
  var run = async action => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setMessage('Working…');
    try {
      await action();
    } catch (e) {
      setMessage(e.message || 'Action failed. Please retry.');
    } finally {
      lock.current = false;
      setBusy(false);
    }
  };
  var email = async () => {
    var creds = JSON.parse(localStorage.getItem('emailCredentials') || '{}');
    if (!recipient.trim()) throw Error('Enter your email address.');
    if (!creds.gmailUser || !creds.gmailPassword) throw Error('Configure Gmail in Settings before sending.');
    var response = await fetch(`${api}/api/email-research`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        email: recipient.trim(),
        subject: `${ticker} — Catalyst ${emailScope === 'all' ? 'all notes' : labels[emailScope]}`,
        content: renderHtml(emailContent(value, sections, emailScope, labels)),
        promptName: 'Improved Catalyst',
        ticker,
        minimal: true,
        smtpConfig: {
          use_gmail: creds.useGmail,
          gmail_user: creds.gmailUser,
          gmail_app_password: creds.gmailPassword,
          from_email: creds.gmailUser
        }
      })
    });
    if (!response.ok) {
      var error = await response.json().catch(() => ({}));
      throw Error(error.error || 'Email failed.');
    }
    setMessage(`Email sent to ${recipient.trim()}.`);
    setEmailScope(null);
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "my-4 rounded-xl border border-amber-500/30 bg-black/10 p-4 sm:p-5 min-w-0",
    "aria-label": "Improved catalyst comparison",
    style: {
      color: 'var(--ink)',
      background: 'var(--surface)',
      borderColor: 'var(--line)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap items-center justify-between gap-2"
  }, /*#__PURE__*/React.createElement("h3", {
    className: "text-base font-semibold"
  }, "Improved catalyst \xB7 comparison"), /*#__PURE__*/React.createElement("span", {
    className: "text-xs opacity-75"
  }, "Original preserved separately")), /*#__PURE__*/React.createElement("p", {
    className: "text-sm leading-relaxed mt-2 opacity-80"
  }, value.workflowMode === 'event' ? 'One event note in your analytical voice. Source attribution and review stay in the private evidence view.' : value.editorialVersion ? 'PM takeaway is the concise investment readout. Detailed note covers the discussion by theme. Q&A follows each question and management answer. Evidence & review holds the private source audit.' : 'Earlier trial format. A new synthesis uses the revised investor-note format; this saved version is preserved.'), value.status === 'failed' ? /*#__PURE__*/React.createElement("p", {
    role: "status",
    className: "mt-3 text-sm"
  }, "The improved note could not finish. Your original is preserved. Re-run the recap to retry.") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2 my-4",
    role: "group",
    "aria-label": "Improved note sections"
  }, Object.entries(labels).map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    onClick: () => setView(id),
    "aria-pressed": view === id,
    style: view === id ? {
      background: 'var(--accent)',
      color: 'var(--on-accent)'
    } : undefined,
    className: `px-3 py-2 text-sm rounded-lg border ${view === id ? 'bg-amber-500 text-black border-amber-500' : 'border-white/20'}`
  }, label))), view === 'quick' && !!value.limitations?.length && /*#__PURE__*/React.createElement("details", {
    className: "mb-4 text-sm"
  }, /*#__PURE__*/React.createElement("summary", {
    className: "cursor-pointer"
  }, "Coverage notes \xB7 ", value.limitations.length), /*#__PURE__*/React.createElement("ul", {
    className: "list-disc pl-5 mt-2"
  }, value.limitations.map((note, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, note)))), view === 'quick' && value.editorialFailure && /*#__PURE__*/React.createElement("details", {
    className: "mb-4 text-sm"
  }, /*#__PURE__*/React.createElement("summary", null, "Why the investor-note draft needs revision"), /*#__PURE__*/React.createElement("ul", null, (value.editorialFailure.findings || []).map((f, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, "Paragraph ", f.block + 1, ": ", f.reason))), /*#__PURE__*/React.createElement("p", null, "The source evidence and original recap are preserved. This draft has not been enabled for sharing.")), view === 'qa' && /*#__PURE__*/React.createElement("button", {
    className: "px-3 py-2 mb-4 rounded-lg border text-sm",
    "aria-pressed": showSpeakers,
    onClick: () => setShowSpeakers(v => !v)
  }, "Show speaker names: ", showSpeakers ? 'On' : 'Off'), /*#__PURE__*/React.createElement("div", {
    className: "prose prose-invert prose-sm max-w-none leading-relaxed break-words",
    style: {
      overflowWrap: 'anywhere',
      fontFamily: 'Calibri,Carlito,Arial,sans-serif',
      fontSize: '11pt'
    },
    dangerouslySetInnerHTML: {
      __html: renderHtml(selected)
    }
  }), view !== 'quick' && /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2 mt-4 border-t border-white/10 pt-4"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy || !shareable.includes(view),
    className: "px-3 py-2 rounded-lg border border-white/20 text-sm",
    onClick: () => run(async () => {
      var doc = new DOMParser().parseFromString(selected, 'text/html');
      doc.querySelectorAll('h1,h2,h3,p,li').forEach(node => node.appendChild(doc.createTextNode('\n\n')));
      var plain = (doc.body.textContent || '').trim();
      if (window.ClipboardItem && navigator.clipboard.write) {
        await navigator.clipboard.write([new ClipboardItem({
          'text/html': new Blob([renderHtml(selected)], {
            type: 'text/html'
          }),
          'text/plain': new Blob([plain], {
            type: 'text/plain'
          })
        })]);
      } else {
        await navigator.clipboard.writeText(plain);
      }
      setMessage('Formatted note copied.');
    })
  }, "Copy ", labels[view]), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !shareable.includes(view),
    className: "px-3 py-2 rounded-lg border border-white/20 text-sm",
    onClick: () => prepareEmail(view)
  }, "Email ", labels[view]), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !shareable.length,
    className: "px-3 py-2 rounded-lg border border-white/20 text-sm",
    onClick: () => prepareEmail('all')
  }, "Email all notes"), view === 'qa' && /*#__PURE__*/React.createElement("button", {
    disabled: busy || !shareable.includes('qa'),
    className: "px-3 py-2 rounded-lg border text-sm",
    onClick: () => run(async () => {
      var blob = new Blob(['<!doctype html><html><head><meta charset="utf-8"><title>Q&amp;A</title></head><body>' + renderHtml(selected) + '</body></html>'], {
        type: 'text/html'
      });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = `${ticker}-QA.html`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setMessage('Q&A saved with the current speaker-name setting.');
    })
  }, "Save Q&A"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !jobId || !!value.editorialVersion && !value.shareMarkdown,
    style: {
      background: 'var(--accent)',
      color: 'var(--on-accent)'
    },
    className: "px-3 py-2 rounded-lg bg-amber-500 text-black text-sm",
    onClick: () => run(async () => {
      var response = await fetch(`${api}/api/catalysts/result/${encodeURIComponent(jobId)}/save?variant=improved`, {
        method: 'POST'
      });
      var body = await response.json();
      if (!response.ok) throw Error(body.error || 'Save failed.');
      setMessage('Improved note saved separately in Research → Catalyst Synthesis.');
    })
  }, value.workflowMode === 'event' ? 'Save event note' : 'Save detailed note')), emailScope && /*#__PURE__*/React.createElement("section", {
    "aria-label": "Email preview",
    className: "mt-4 border rounded-lg p-4 min-w-0"
  }, /*#__PURE__*/React.createElement("h4", {
    className: "font-semibold"
  }, "Email ", emailScope === 'all' ? 'all notes' : labels[emailScope]), /*#__PURE__*/React.createElement("p", {
    className: "text-sm my-2"
  }, emailScope === 'all' ? shareable.map(k => labels[k]).join(', ') : labels[emailScope], ". Private evidence and review details are excluded."), /*#__PURE__*/React.createElement("label", {
    className: "block text-sm"
  }, "To", /*#__PURE__*/React.createElement("input", {
    type: "email",
    value: recipient,
    onChange: e => setRecipient(e.target.value),
    className: "block w-full border rounded p-2 my-2",
    style: {
      color: 'var(--ink)',
      background: 'var(--surface)'
    }
  })), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Preview email"), /*#__PURE__*/React.createElement("div", {
    style: {
      maxHeight: 420,
      overflowY: 'auto',
      overflowWrap: 'anywhere'
    },
    dangerouslySetInnerHTML: {
      __html: renderHtml(emailContent(value, sections, emailScope, labels))
    }
  })), /*#__PURE__*/React.createElement("div", {
    className: "flex flex-wrap gap-2 mt-3"
  }, /*#__PURE__*/React.createElement("button", {
    className: "border rounded px-3 py-2",
    disabled: busy || !recipient.trim(),
    onClick: () => run(email)
  }, "Send email"), /*#__PURE__*/React.createElement("button", {
    className: "border rounded px-3 py-2",
    disabled: busy,
    onClick: () => setEmailScope(null)
  }, "Cancel"))), /*#__PURE__*/React.createElement("p", {
    role: "status",
    "aria-live": "polite",
    className: "text-sm mt-2"
  }, message)));
}