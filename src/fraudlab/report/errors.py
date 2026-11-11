"""Errors shared by report rendering and publication exports."""


class ReportError(RuntimeError):
    """A run folder is missing, unusable or not approved for publication."""


class RestrictedDatasetError(ReportError):
    """Publication rights have not been established for a dataset's artifacts."""
