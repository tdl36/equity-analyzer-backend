# Catalyst Q&A and email trial

Catalyst-only addition; the existing Summary workflow is untouched.

- Distinct transcript questions and management answers have a separate Q&A view in source order, including the opening moderator question and substantive follow-ups.
- Generation selects original passage IDs, with a separate model check for fidelity and missing questions. These checks do not independently verify management claims.
- Documents without a distinct Q&A return not-applicable; failed Q&A does not discard the other notes. Failed Q&A is excluded from email.
- The Catalyst viewer offers current-tab or combined emails, recipient review and explicit Send. It uses Charlie's existing email Settings and endpoint. Private evidence is excluded.
- The standalone local review uses a loopback server because file previews cannot access Charlie's saved email Settings. It accepts Gmail credentials in memory for an explicit send, or downloads a MIME email draft. Credentials are cleared on dialog close. No messages are sent by generating the review.

Local preview server: `scripts/catalyst_review_server.py <review.html> --port 8794`.
Only the specified HTML is served. Email forwarding requires same-origin and a per-process token and uses one fixed Charlie endpoint. No keys are embedded in artifacts.

Validation: standalone unit scripts only (do not run pytest against the live research database), email section isolation, and browser email requests intercepted in testing. Actual SMTP delivery is not tested automatically.

This is local development, not a production rollout.
