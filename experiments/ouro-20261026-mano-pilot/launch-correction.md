# Aborted extension launch

The first extension launch was aborted: a configuration-generation command
outlived its initial tool wait, so subsequent staging failed before the new
configuration files existed. The launcher nevertheless started once files became
available. No metrics from this launch are accepted as reproduction evidence.
Its ignored runtime directory is retained for audit.

Correction: explicitly wait for command completion before dependent operations;
strengthen the pre-run cleanliness check to reject untracked files as well as
tracked modifications. Commit the complete configuration before a new attempt
in a distinct output directory. The model/data/optimizer implementation is unchanged.
