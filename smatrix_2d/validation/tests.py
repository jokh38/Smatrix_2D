"""Validation test suite for operator-factorized transport."""

import numpy as np

from smatrix_2d.core.grid import PhaseSpaceGrid2D
from smatrix_2d.core.state import create_initial_state
from smatrix_2d.core.materials import MaterialProperties2D
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.transport.transport_step import (
    FirstOrderSplitting,
)
from smatrix_2d.operators.angular_scattering import (
    AngularScatteringOperator,
)
from smatrix_2d.operators.spatial_streaming import (
    SpatialStreamingOperator,
    BackwardTransportMode,
)
from smatrix_2d.operators.energy_loss import EnergyLossOperator

from smatrix_2d.validation.metrics import (
    compute_gamma_pass_rate,
    check_rotational_invariance,
)


class TransportValidator:
    """Comprehensive validation test suite.

    Tests:
    - Operator conservation (isolated)
    - Positivity preservation
    - Vacuum transport
    - Rotational invariance (ray effect)
    """

    def __init__(
        self,
        grid: PhaseSpaceGrid2D,
        material: MaterialProperties2D,
        constants: PhysicsConstants2D,
    ):
        """Initialize validator.

        Args:
            grid: Phase space grid
            material: Material properties
            constants: Physics constants
        """
        self.grid = grid
        self.material = material
        self.constants = constants

        # Create operators
        self.A_theta = AngularScatteringOperator(grid, material, constants)
        self.A_stream = SpatialStreamingOperator(
            grid, constants, BackwardTransportMode.HARD_REJECT
        )
        self.A_E = EnergyLossOperator(grid)

        self.transport = FirstOrderSplitting(
            self.A_theta, self.A_stream, self.A_E
        )

    def check_conservation(
        self,
        psi: np.ndarray,
        psi_out: np.ndarray,
        tolerance: float = 1e-10,
    ) -> bool:
        """Check operator conservation.

        sum(psi_out) = sum(psi) within tolerance

        Args:
            psi: Input state
            psi_out: Output state
            tolerance: Maximum allowed relative error

        Returns:
            True if conservation holds
        """
        sum_in = np.sum(psi)
        sum_out = np.sum(psi_out)

        if sum_in < tolerance:
            return True

        relative_error = abs(sum_out - sum_in) / sum_in

        return relative_error <= tolerance

    def check_positivity(
        self,
        psi: np.ndarray,
        epsilon: float = 1e-12,
    ) -> bool:
        """Check positivity preservation.

        psi >= 0 everywhere

        Args:
            psi: State to check
            epsilon: Numerical epsilon

        Returns:
            True if all values non-negative
        """
        return np.all(psi >= -epsilon)

    def test_vacuum_transport(
        self,
        n_steps: int = 100,
        oblique_angle: float = np.pi / 4.0,
    ) -> dict:
        """Test vacuum transport (no scattering, no energy loss).

        Shoots beam at 45 degrees, verifies straight-line motion.

        Args:
            n_steps: Number of transport steps
            oblique_angle: Beam angle [rad]

        Returns:
            Test results dict
        """
        # Create vacuum material
        from smatrix_2d.core.materials import MaterialProperties2D

        vacuum_mat = MaterialProperties2D(
            name='vacuum',
            rho=0.0,
            X0=1e10,
            Z=1.0,
            A=1.0,
            I_excitation=1e-3,
        )

        # Vacuum operators
        A_theta_vac = AngularScatteringOperator(self.grid, vacuum_mat, self.constants)
        A_stream_vac = SpatialStreamingOperator(
            self.grid, self.constants, BackwardTransportMode.HARD_REJECT
        )
        A_E_vac = EnergyLossOperator(self.grid)

        transport_vac = FirstOrderSplitting(A_theta_vac, A_stream_vac, A_E_vac)

        # Initial state at center, moving at oblique angle
        state = create_initial_state(
            self.grid,
            x_init=self.grid.x_centers[len(self.grid.x_centers) // 2],
            z_init=0.0,
            theta_init=oblique_angle,
            E_init=50.0,
            initial_weight=1.0,
        )

        # Stopping power function (returns 0 for vacuum)
        def stopping_power_vac(E):
            return 0.0

        # Track centroid
        x_centroid = []
        z_centroid = []

        for step in range(n_steps):
            state = transport_vac.apply(state, stopping_power_vac)

            # Compute centroid
            total = np.sum(state.psi)

            if total > 1e-12:
                x_c = np.sum(state.psi * self.grid.x_centers[np.newaxis, :]) / total
                z_c = np.sum(state.psi * self.grid.z_centers[:, np.newaxis]) / total

                x_centroid.append(x_c)
                z_centroid.append(z_c)

        # Compute centroid drift
        x_drift = np.std(x_centroid)
        z_drift = np.std(z_centroid)

        # Expected trajectory
        v_x = np.cos(oblique_angle)
        v_z = np.sin(oblique_angle)

        # Final position
        x_expected = self.grid.x_centers[len(self.grid.x_centers) // 2] + n_steps * \
            self.grid.delta_z * v_x
        z_expected = n_steps * self.grid.delta_z * v_z

        x_final = x_centroid[-1] if x_centroid else 0.0
        z_final = z_centroid[-1] if z_centroid else 0.0

        return {
            'x_drift': x_drift,
            'z_drift': z_drift,
            'x_error': abs(x_final - x_expected),
            'z_error': abs(z_final - z_expected),
            'final_weight': state.total_weight(),
            'passed': x_drift < 0.001 and z_drift < 0.001,
        }

    def test_rotational_invariance(
        self,
        angle_a: float = 0.0,
        angle_b: float = np.pi / 4.0,
        n_steps: int = 50,
    ) -> dict:
        """Test rotational invariance to detect ray effect.

        Runs identical beam at two angles, rotates back, compares.

        Args:
            angle_a: First beam angle [rad]
            angle_b: Second beam angle [rad]
            n_steps: Number of transport steps

        Returns:
            Test results dict with gamma metrics
        """
        # Run Case A
        state_a = create_initial_state(
            self.grid,
            x_init=self.grid.x_centers[len(self.grid.x_centers) // 2],
            z_init=0.0,
            theta_init=angle_a,
            E_init=50.0,
            initial_weight=1.0,
        )

        # Run transport
        def stopping_power(E):
            return 2.0e-3  # Simple approximation

        for _ in range(n_steps):
            state_a = self.transport.apply(state_a, stopping_power)

        dose_a = state_a.deposited_energy

        # Run Case B
        state_b = create_initial_state(
            self.grid,
            x_init=self.grid.x_centers[len(self.grid.x_centers) // 2],
            z_init=0.0,
            theta_init=angle_b,
            E_init=50.0,
            initial_weight=1.0,
        )

        for _ in range(n_steps):
            state_b = self.transport.apply(state_b, stopping_power)

        dose_b = state_b.deposited_energy

        # Compute rotational invariance
        rotation_angle = angle_b - angle_a
        l2_error, linf_error = check_rotational_invariance(
            dose_a, dose_b, rotation_angle,
            self.grid.x_centers, self.grid.z_centers,
        )

        gamma_rate = compute_gamma_pass_rate(
            dose_a, dose_b,
            self.grid.x_centers, self.grid.z_centers,
            dose_threshold=2.0,
            distance_threshold=1.0,
        )

        return {
            'l2_error': l2_error,
            'linf_error': linf_error,
            'gamma_pass_rate': gamma_rate,
            'passed': l2_error <= 0.02 and linf_error <= 0.05,
        }

    def run_all_tests(self) -> dict:
        """Run complete validation suite.

        Returns:
            Comprehensive test results
        """
        results = {}

        results['vacuum'] = self.test_vacuum_transport()
        results['rotational_invariance'] = self.test_rotational_invariance()

        passed_count = sum(1 for r in results.values() if r.get('passed', False))
        total_count = len(results)

        results['summary'] = {
            'total_tests': total_count,
            'passed_tests': passed_count,
            'pass_rate': passed_count / total_count if total_count > 0 else 0.0,
        }

        return results
