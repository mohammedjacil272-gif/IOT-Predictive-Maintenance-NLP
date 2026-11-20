# NLP workflow completion

The NLP-only path preserves original text, normalizes it, extracts rule-based entities with source offsets, and returns heuristic severity with matched terms. Components, faults, symptoms, actions and measurements remain rule labels; severity is explicitly heuristic, not a trained prediction. Empty or invalid text is handled without transformer substitution. Negated severity terms are excluded by the existing negation rules.
