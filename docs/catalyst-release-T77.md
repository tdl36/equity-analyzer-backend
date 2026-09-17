# T77 Catalyst investor notes rollout

User authorized production deployment on September 17, 2026. Summary Lab remains the committed T76 implementation; original Summary generation is unchanged.

New catalyst runs retain the original recap and generate a separate improved note. Transcript sources produce PM takeaway, Detailed note and Q&A (when present). Event folders produce My takeaway, What happened, Why it matters, What remains unproven and What I’m watching next. No broker names or investment ratings in the public event note; evidence remains private. Email current/all notes and separate improved-note save are available. Q&A speaker visibility also controls email and Q&A export.

Existing outputs are not backfilled. Rerunning a catalyst incurs additional model calls. Restart the Mac agent after the backend release to activate iCloud generation. No acceptance or thesis changes are automatic.

Validation: 38 safe Catalyst unit tests and 40 frontend tests passed. Local MRK and ELV previews were browser-tested previously. MRK final preview required editorial intervention before source review; this is not proof of fully automatic editorial reliability. Source matching and model review can still miss omissions or interpretation errors. Filename-based event routing has an explicit source-part override but no user-facing mode selector yet.

This rollout supersedes the local-only deployment status in earlier trial logs; those logs preserve the actual development history.
