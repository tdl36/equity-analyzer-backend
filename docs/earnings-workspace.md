# Earnings & evidence workspace

Research desk → Earnings & evidence uses the existing pending and failed analyst activity endpoints. It combines active events with the latest 100 failures, deduplicates IDs, and limits the view to earnings recaps/takeaways with event topics. It neither changes activity state nor triggers paid research. Approvals, source selection and regeneration remain in Analyst team.

The source register comes from output.sourceFiles, not a live filesystem scan. Coverage categories are filename heuristics and explicitly distinguish “not identified” from missing files. Counts are reconciled against fileCount when available. Model-generated contribution records are labeled as such; they are not independent citation checks. Failed regenerations retain their prior draft with an explicit label. Unknown extraction completeness, numerical accuracy and page citations remain unchecked.

This first release provides event navigation, visible failures, source coverage and readable synthesis with a saved-thesis comparison action. It does not yet implement a live folder inventory, independent claim-level validation, page viewers or an automatically generated thesis delta. Those require additional backend evidence capture and generation work. Approved events stay in the existing archive.

Validation: frontend model tests cover queued vs draft states, retained output after failures, source deduplication and coverage reconciliation. Live browser verification reads existing production activities without altering them.
