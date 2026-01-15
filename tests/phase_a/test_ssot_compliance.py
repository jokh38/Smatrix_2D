"""
Test Suite for SSOT (Single Source of Truth) Compliance

This module implements V-CFG-001: Static analysis for hardcoded defaults.

Purpose:
    Ensure all default configuration values are sourced from smatrix_2d/config/defaults.py
    and not hardcoded throughout the codebase.

Strategy:
    1. Parse Python files using AST to find hardcoded float/int literals
    2. Match against known default values from defaults.py
    3. Check if the file imports from defaults.py (allowlist)
    4. Report violations with file locations and suggested fixes

Rationale:
    Hardcoded defaults scattered across the codebase lead to:
    - Configuration drift (different files using different defaults)
    - Maintenance burden (updates require changes in multiple places)
    - Subtle bugs (inconsistent defaults causing unexpected behavior)
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest


# =============================================================================
# Known Default Values from SSOT
# =============================================================================

# Files that are exempt from SSOT checks (e.g., defaults.py itself, test utilities)
SSOT_EXEMPT_FILES = {
    'smatrix_2d/config/defaults.py',  # SSOT file itself
}

# Context patterns that should not be flagged (false positives)
SSOT_EXEMPT_PATTERNS = {
    # Boolean-like constants
    r'CUPY_AVAILABLE',  # Module availability flag
    r'GPU_AVAILABLE',   # Module availability flag
    # Feature flags or state variables
    r'conservation_valid',  # State flag, not a default
    r'is_.*',              # State flags
    r'has_.*',             # State flags
    # Profiling/debugging variables
    r'.*profile.*',
    r'.*debug.*',
    # Tiling parameters (not config defaults)
    r'tile_size.*',
    r'halo_size',
    r'total_tiles',
    # Loop counters and iteration variables
    r'i_', r'j_', r'k_',
    r'iter',
    r'count',
}

# Mapping of hardcoded values that should use defaults.py constants
# Format: value -> [(constant_name, description, context_filter)]
SSOT_PATTERNS = {
    # Energy defaults (CRITICAL - these cause bugs)
    0.0: [
        ("LEGACY_E_MIN", "DEPRECATED: Old E_min, causes E_min/E_cutoff conflict",
         lambda ctx, name: ctx == 'constant' and name == 'LEGACY_E_MIN'),
        ("DEFAULT_THETA_MIN", "Minimum angle (degrees)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_THETA_MIN'),
    ],
    1.0: [
        ("DEFAULT_E_MIN", "Minimum energy (MeV)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_E_MIN'),
        ("DEFAULT_DELTA_X", "Spatial resolution x (mm)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_DELTA_X'),
        ("DEFAULT_DELTA_Z", "Spatial resolution z (mm)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_DELTA_Z'),
        ("DEFAULT_DELTA_S", "Transport step size (mm)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_DELTA_S'),
        ("DEFAULT_WATER_DENSITY", "Water density (g/cm³)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_WATER_DENSITY'),
    ],
    2.0: [
        ("DEFAULT_E_CUTOFF", "Energy cutoff threshold (MeV)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_E_CUTOFF'),
    ],

    # Spatial defaults
    50.0: [
        ("DEFAULT_SPATIAL_HALF_SIZE", "Spatial domain half-width (mm)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_SPATIAL_HALF_SIZE'),
    ],

    # Grid dimensions (integers)
    100: [
        ("DEFAULT_NX", "Number of x bins",
         lambda ctx, name: ctx in ['constant', 'field_default', 'function_default'] and name in ['Nx', 'nx']),
        ("DEFAULT_NZ", "Number of z bins",
         lambda ctx, name: ctx in ['constant', 'field_default', 'function_default'] and name in ['Nz', 'nz']),
        ("DEFAULT_NE", "Number of energy bins",
         lambda ctx, name: ctx in ['constant', 'field_default', 'function_default'] and name in ['Ne', 'ne']),
        ("DEFAULT_MAX_STEPS", "Maximum number of transport steps",
         lambda ctx, name: ctx in ['constant', 'field_default', 'function_default'] and 'max_step' in name.lower()),
    ],

    # Angular defaults
    180: [
        ("DEFAULT_NTHETA", "Number of angular bins",
         lambda ctx, name: ctx in ['constant', 'field_default', 'function_default'] and name in ['Ntheta', 'ntheta', 'n_theta']),
    ],
    180.0: [
        ("DEFAULT_THETA_MAX", "Maximum angle (degrees)",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_THETA_MAX'),
    ],

    # Numerical defaults
    1e-10: [
        ("DEFAULT_WEIGHT_THRESHOLD", "Weight threshold for particle tracking",
         lambda ctx, name: ctx == 'constant' and 'weight' in name.lower()),
    ],
    0.01: [
        ("DEFAULT_BETA_SQ_MIN", "Minimum beta squared for Highland formula",
         lambda ctx, name: ctx == 'constant' and 'beta' in name.lower()),
    ],

    # Sigma bucket defaults
    10: [
        ("DEFAULT_N_BUCKETS", "Number of sigma buckets",
         lambda ctx, name: ctx in ['constant', 'field_default', 'function_default'] and ('bucket' in name.lower() or 'n_bucket' in name.lower())),
    ],
    45.0: [
        ("DEFAULT_THETA_CUTOFF_DEG", "Sigma cutoff for angular scattering",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_THETA_CUTOFF_DEG'),
    ],

    # GPU kernel defaults
    256: [
        ("DEFAULT_BLOCK_SIZE_1D", "CUDA block size for 1D kernels",
         lambda ctx, name: ctx == 'constant' and 'block' in name.lower() and '1d' in name.lower()),
    ],

    # Transport defaults
    1: [
        ("DEFAULT_SUB_STEPS", "Number of sub-steps per operator",
         lambda ctx, name: ctx in ['constant', 'field_default', 'function_default'] and 'sub_step' in name.lower()),
    ],

    # Material defaults
    36.08: [
        ("DEFAULT_WATER_RADIATION_LENGTH", "Water radiation length (g/cm²)",
         lambda ctx, name: ctx == 'constant' and 'radiation' in name.lower()),
    ],
    75.0: [
        ("DEFAULT_WATER_MEAN_EXCITATION_ENERGY", "Water mean excitation energy (MeV)",
         lambda ctx, name: ctx == 'constant' and 'excitation' in name.lower()),
    ],
    7.42: [
        ("DEFAULT_WATER_Z", "Effective Z for liquid water",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_WATER_Z'),
    ],
    18.0: [
        ("DEFAULT_WATER_A", "Effective A for liquid water",
         lambda ctx, name: ctx == 'constant' and name == 'DEFAULT_WATER_A'),
    ],
}


# =============================================================================
# AST Analysis Utilities
# =============================================================================

class SSOTVisitor(ast.NodeVisitor):
    """
    AST visitor to find hardcoded default values that should use SSOT constants.

    Detects:
    - Function default arguments with hardcoded values
    - Dataclass field defaults with hardcoded values
    - Variable assignments with hardcoded values
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Dict] = []
        self.imports_defaults: Set[str] = set()

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track imports from defaults.py for allowlisting."""
        if node.module and "defaults" in node.module:
            for alias in node.names:
                self.imports_defaults.add(alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function argument defaults for hardcoded values."""
        # Get parameter names (excluding ones without defaults)
        params_no_default = len(node.args.args) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            # Get the corresponding parameter name
            param_index = params_no_default + i
            if param_index < len(node.args.args):
                param_name = node.args.args[param_index].arg
                self._check_value(default, "function_default", param_name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Check annotated assignments (e.g., dataclass fields) for hardcoded values."""
        if node.value:
            self._check_value(node.value, "field_default", getattr(node.target, "id", "unknown"))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Check regular assignments for hardcoded values."""
        if node.value:
            # Check if this is a constant-like assignment (all caps name)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Flag ALL_CAPS variable assignments
                    if name.isupper() and '_' in name:
                        self._check_value(node.value, "constant", name)
        self.generic_visit(node)

    def _check_value(self, node: ast.AST, context: str, name: str):
        """Check if a node contains a hardcoded SSOT value."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                self._check_numeric_value(node.value, context, name, node.lineno)
        elif isinstance(node, ast.Num):  # Deprecated in Python 3.14, but keep for compatibility
            self._check_numeric_value(node.n, context, name, node.lineno)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                self._check_value(elt, context, name)

    def _check_numeric_value(self, value: float, context: str, name: str, lineno: int):
        """Check if a numeric value matches an SSOT pattern."""
        # Check if name matches any exempt pattern (false positives)
        for pattern in SSOT_EXEMPT_PATTERNS:
            if re.match(pattern, name):
                return

        # Convert to float for comparison
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return

        # Check against SSOT patterns
        if value_float in SSOT_PATTERNS:
            for const_name, description, context_filter in SSOT_PATTERNS[value_float]:
                # Apply context filter to avoid false positives
                if context_filter(context, name):
                    self.violations.append({
                        'line': lineno,
                        'value': value_float,
                        'context': context,
                        'name': name,
                        'suggested_constant': const_name,
                        'description': description,
                    })
                    # Only report first match per value
                    break


# =============================================================================
# File Scanning Utilities
# =============================================================================

def get_python_files(root_dir: Path) -> List[Path]:
    """Get all Python files in the project (excluding tests and __pycache__)."""
    python_files = []
    for path in root_dir.rglob("*.py"):
        # Skip test files (they can have test-specific hardcoded values)
        if "tests" in path.parts:
            continue
        # Skip __pycache__
        if "__pycache__" in path.parts:
            continue
        # Skip .tox, .pytest_cache, etc.
        if any(part.startswith('.') for part in path.parts):
            continue

        # Check if file should be exempt
        path_str = str(path)
        if any(exempt in path_str for exempt in SSOT_EXEMPT_FILES):
            continue

        python_files.append(path)
    return python_files


def scan_file_for_violations(filepath: Path) -> Tuple[List[Dict], Set[str]]:
    """Scan a single Python file for SSOT violations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        # Skip files that can't be parsed
        return [], set()

    visitor = SSOTVisitor(str(filepath))
    visitor.visit(tree)
    return visitor.violations, visitor.imports_defaults


def should_allowlist_violation(violation: Dict, imports: Set[str]) -> bool:
    """
    Determine if a violation should be allowlisted.

    Allowlist if:
    1. The file imports the suggested constant
    2. The context is a test file (already filtered)
    """
    suggested_const = violation['suggested_constant']
    return suggested_const in imports


# =============================================================================
# Test Cases
# =============================================================================

class TestSSOTCompliance:
    """Test suite for SSOT compliance across the codebase."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.project_root = Path("/workspaces/Smatrix_2D")
        self.source_dir = self.project_root / "smatrix_2d"

    def test_no_hardcoded_energy_defaults_in_source(self):
        """
        V-CFG-001.1: Ensure energy defaults (E_min, E_cutoff) are not hardcoded.

        Critical test: Hardcoded energy defaults cause configuration drift
        and subtle bugs. All energy defaults must come from defaults.py.
        """
        violations_by_file = {}
        python_files = get_python_files(self.source_dir)

        for filepath in python_files:
            violations, imports = scan_file_for_violations(filepath)

            # Filter out allowlisted violations
            unallowed = [
                v for v in violations
                if not should_allowlist_violation(v, imports)
                # Focus on energy-related violations
                and 'E_MIN' in v['suggested_constant'] or 'E_CUTOFF' in v['suggested_constant']
            ]

            if unallowed:
                violations_by_file[str(filepath.relative_to(self.project_root))] = unallowed

        if violations_by_file:
            # Format failure message
            msg = ["\n=== SSOT VIOLATIONS: Hardcoded Energy Defaults ===\n"]
            for filepath, viols in violations_by_file.items():
                msg.append(f"\nFile: {filepath}")
                for v in viols:
                    msg.append(f"  Line {v['line']}: {v['context']} '{v['name']}'")
                    msg.append(f"    Hardcoded: {v['value']} -> Should use: {v['suggested_constant']}")
                    msg.append(f"    Description: {v['description']}")

            pytest.fail("\n".join(msg))

    def test_no_hardcoded_spatial_defaults_in_source(self):
        """
        V-CFG-001.2: Ensure spatial defaults are not hardcoded.

        Spatial defaults (SPATIAL_HALF_SIZE, NX, NZ) should come from defaults.py.
        """
        violations_by_file = {}
        python_files = get_python_files(self.source_dir)

        for filepath in python_files:
            violations, imports = scan_file_for_violations(filepath)

            # Filter spatial violations
            unallowed = [
                v for v in violations
                if not should_allowlist_violation(v, imports)
                and any(x in v['suggested_constant']
                       for x in ['SPATIAL_HALF_SIZE', 'DEFAULT_NX', 'DEFAULT_NZ'])
            ]

            if unallowed:
                violations_by_file[str(filepath.relative_to(self.project_root))] = unallowed

        if violations_by_file:
            msg = ["\n=== SSOT VIOLATIONS: Hardcoded Spatial Defaults ===\n"]
            for filepath, viols in violations_by_file.items():
                msg.append(f"\nFile: {filepath}")
                for v in viols:
                    msg.append(f"  Line {v['line']}: {v['context']} '{v['name']}'")
                    msg.append(f"    Hardcoded: {v['value']} -> Should use: {v['suggested_constant']}")

            pytest.fail("\n".join(msg))

    def test_no_hardcoded_grid_defaults_in_source(self):
        """
        V-CFG-001.3: Ensure grid dimension defaults are not hardcoded.

        Grid dimensions (NE, NTHETA) should come from defaults.py.
        """
        violations_by_file = {}
        python_files = get_python_files(self.source_dir)

        for filepath in python_files:
            violations, imports = scan_file_for_violations(filepath)

            # Filter grid violations
            unallowed = [
                v for v in violations
                if not should_allowlist_violation(v, imports)
                and any(x in v['suggested_constant']
                       for x in ['DEFAULT_NE', 'DEFAULT_NTHETA'])
            ]

            if unallowed:
                violations_by_file[str(filepath.relative_to(self.project_root))] = unallowed

        if violations_by_file:
            msg = ["\n=== SSOT VIOLATIONS: Hardcoded Grid Defaults ===\n"]
            for filepath, viols in violations_by_file.items():
                msg.append(f"\nFile: {filepath}")
                for v in viols:
                    msg.append(f"  Line {v['line']}: {v['context']} '{v['name']}'")
                    msg.append(f"    Hardcoded: {v['value']} -> Should use: {v['suggested_constant']}")

            pytest.fail("\n".join(msg))

    def test_transport_api_uses_defaults(self):
        """
        V-CFG-001.4: Ensure transport API functions use defaults.py.

        Specific check for smatrix_2d/transport/api.py and transport.py
        which commonly have hardcoded grid dimension defaults.
        """
        api_file = self.source_dir / "transport" / "api.py"
        transport_file = self.source_dir / "transport" / "transport.py"

        violations_summary = []

        for filepath in [api_file, transport_file]:
            if not filepath.exists():
                continue

            violations, imports = scan_file_for_violations(filepath)

            # Check for Nx=100, Nz=100, Ne=100 in function signatures
            for v in violations:
                if v['context'] == 'function_default' and v['name'] == 'create_simulation':
                    if not should_allowlist_violation(v, imports):
                        violations_summary.append(
                            f"{filepath.name}:{v['line']} - "
                            f"create_simulation() has hardcoded {v['name']}={v['value']}, "
                            f"should use {v['suggested_constant']}"
                        )

        if violations_summary:
            pytest.fail(
                "\n=== TRANSPORT API SSOT VIOLATIONS ===\n" +
                "\n".join(violations_summary)
            )

    def test_summary_of_all_violations(self):
        """
        V-CFG-001.5: Generate comprehensive summary of all SSOT violations.

        This test provides a complete report of all violations across the codebase,
        making it easy to identify and fix issues systematically.
        """
        violations_by_file = {}
        python_files = get_python_files(self.source_dir)

        total_violations = 0
        allowlisted_violations = 0

        for filepath in python_files:
            violations, imports = scan_file_for_violations(filepath)

            # Separate violations
            unallowed = []
            allowed = []

            for v in violations:
                if should_allowlist_violation(v, imports):
                    allowed.append(v)
                else:
                    unallowed.append(v)

            if unallowed:
                violations_by_file[str(filepath.relative_to(self.project_root))] = unallowed

            total_violations += len(violations)
            allowlisted_violations += len(allowed)

        # Print summary report
        print("\n" + "="*80)
        print("SSOT COMPLIANCE SUMMARY")
        print("="*80)

        print(f"\nFiles scanned: {len(python_files)}")
        print(f"Total potential violations: {total_violations}")
        print(f"Allowlisted (imports from defaults.py): {allowlisted_violations}")
        print(f"Unallowed violations: {total_violations - allowlisted_violations}")

        if violations_by_file:
            print(f"\nFiles with violations: {len(violations_by_file)}")
            print("\nViolations by file:")
            print("-"*80)

            for filepath, viols in sorted(violations_by_file.items()):
                print(f"\n{filepath} ({len(viols)} violations):")
                for v in viols:
                    print(f"  Line {v['line']}: {v['value']} in {v['context']} '{v['name']}'")
                    print(f"    -> Should use: {v['suggested_constant']}")

            print("\n" + "="*80)

            # Fail the test to enforce compliance
            pytest.fail(
                f"\nFound {total_violations - allowlisted_violations} SSOT violations. "
                "Fix by importing constants from smatrix_2d.config.defaults"
            )
        else:
            print("\nNo violations found - SSOT compliance achieved!")
            print("="*80)


# =============================================================================
# Standalone Analysis (for debugging)
# =============================================================================

def analyze_ssot_compliance():
    """
    Standalone function to analyze SSOT compliance without pytest.

    Usage:
        python -c "from tests.phase_a.test_ssot_compliance import analyze_ssot_compliance; analyze_ssot_compliance()"
    """
    project_root = Path("/workspaces/Smatrix_2D")
    source_dir = project_root / "smatrix_2d"

    print("Analyzing SSOT compliance...")
    print("="*80)

    python_files = get_python_files(source_dir)
    violations_by_file = {}

    for filepath in python_files:
        violations, imports = scan_file_for_violations(filepath)

        unallowed = [
            v for v in violations
            if not should_allowlist_violation(v, imports)
        ]

        if unallowed:
            violations_by_file[str(filepath.relative_to(project_root))] = unallowed

    # Print summary
    print(f"\nFiles scanned: {len(python_files)}")
    print(f"Files with violations: {len(violations_by_file)}")
    print("\nViolations:")

    for filepath, viols in sorted(violations_by_file.items()):
        print(f"\n{filepath}:")
        for v in viols:
            print(f"  Line {v['line']}: {v['value']} -> {v['suggested_constant']}")

    print("\n" + "="*80)

    return violations_by_file


if __name__ == "__main__":
    analyze_ssot_compliance()
