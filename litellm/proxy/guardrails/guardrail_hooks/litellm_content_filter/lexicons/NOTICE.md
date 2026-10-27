# Sensitive-lexicon

Source: https://github.com/konsheng/Sensitive-lexicon

Pinned revision: d967c30b053fa40b06c5a0dddf0be493f2dfae46

The 17 files under Vocabulary and LICENSE are unmodified upstream data.
Names describe the upstream categories, not an official classification or an
endorsement of their contents. Loading removes blank lines and duplicate entries
within each file. Terms are literal substrings, not regexes or semantic rules.
No automatic downloads or updates occur at runtime

AgentOS curation (2026-10-27-r1): excluded_terms.txt removes reviewed neutral
standalone entries, case-insensitively, across all dictionaries. It covers common
places, civic terms, employment/commerce, anatomy, religion and ordinary technical
abbreviations. This is an entry exclusion, not a message allowlist: longer risk
phrases containing these words remain active. Preview YAML and runtime consume
the same curated terms. Empty dictionaries are omitted from new selections;
existing IDs remain readable. The upstream files remain unchanged for auditing.
This is a first curation pass, not a guarantee that every remaining entry is
appropriate for every business context.

Revision 2026-10-27-r2 additionally excludes bare numbers, formatted contact
numbers, legacy regex/template fragments and replacement mappings. These are
whole-entry checks, not substring deletions. Reviewed short fragments and neutral
technical words are recorded in excluded_terms.txt. Numeric risk phrases and
domains are not removed by the numeric checks. Personal-information detection
belongs to dedicated format rules, not a historical list of concrete numbers.
exclusion_reason() records why an entry is omitted; upstream data stays intact.
