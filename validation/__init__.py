"""
Validation Package for Smatrix_2D

This package provides validation tools for the GPU-only simulation:
- Golden snapshot comparison (regression testing)
- NIST range validation (physics correctness)

Import Policy:
    from validation.compare import GoldenSnapshot, compare_results
    from validation.nist_validation import validate_nist_range

DO NOT use: from validation import *
"""

from validation.compare import (
    GoldenSnapshot,
    ComparisonResult,
    ToleranceConfig,
    compare_results,
    create_snapshot,
)

from validation.nist_validation import (
    NISTRangeData,
    RangeValidationResult,
    RangeTableResult,
    NISTRangeValidator,
    validate_nist_range,
    validate_range_table,
    calculate_range_from_dose,
)

__all__ = [
    # Golden snapshot comparison
    "GoldenSnapshot",
    "ComparisonResult",
    "ToleranceConfig",
    "compare_results",
    "create_snapshot",
    # NIST validation
    "NISTRangeData",
    "RangeValidationResult",
    "RangeTableResult",
    "NISTRangeValidator",
    "validate_nist_range",
    "validate_range_table",
    "calculate_range_from_dose",
]
