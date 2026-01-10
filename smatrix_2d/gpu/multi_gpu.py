"""Multi-GPU transport implementation using spatial domain decomposition.

Splits the spatial domain (z-direction) across multiple GPUs with halo exchange.
"""

import numpy as np
from typing import List, Tuple, Optional
import cupy as cp

from smatrix_2d.gpu.kernels import GPUTransportStep, AccumulationMode


class MultiGPUTransportStep:
    """Multi-GPU transport step using spatial domain decomposition.

    Decomposes the domain in the z-direction (depth) across multiple GPUs.
    Each GPU handles a local portion of the domain with halo regions for
    boundary communication.
    """

    def __init__(
        self,
        Ne: int,
        Ntheta: int,
        Nz: int,
        Nx: int,
        num_gpus: int = 4,
        accumulation_mode: str = AccumulationMode.FAST,
        delta_x: float = 1.0,
        delta_z: float = 1.0,
        halo_depth: int = 2,
    ):
        """Initialize multi-GPU transport step.

        Args:
            Ne: Number of energy bins
            Ntheta: Number of angular bins
            Nz: Number of depth bins (total across all GPUs)
            Nx: Number of lateral bins
            num_gpus: Number of GPUs to use (default: 4)
            accumulation_mode: 'fast' or 'deterministic'
            delta_x: Lateral grid spacing [mm]
            delta_z: Depth grid spacing [mm]
            halo_depth: Number of halo bins for boundary exchange
        """
        if not cp.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.Ne = Ne
        self.Ntheta = Ntheta
        self.Nz_total = Nz
        self.Nx = Nx
        self.num_gpus = num_gpus
        self.accumulation_mode = accumulation_mode
        self.delta_x = delta_x
        self.delta_z = delta_z
        self.halo_depth = halo_depth

        # Check that we have enough GPUs
        available_gpus = cp.cuda.runtime.getDeviceCount()
        if available_gpus < num_gpus:
            raise RuntimeError(f"Requested {num_gpus} GPUs but only {available_gpus} available")

        # Decompose domain in z-direction
        self.z_bins_per_gpu = Nz // num_gpus
        self.z_remainder = Nz % num_gpus

        # Calculate local domain sizes for each GPU
        self.local_Nz_list = []
        self.z_offsets = []
        z_offset = 0
        for i in range(num_gpus):
            # Distribute remainder bins to first few GPUs
            local_Nz = self.z_bins_per_gpu + (1 if i < self.z_remainder else 0)
            self.local_Nz_list.append(local_Nz)
            self.z_offsets.append(z_offset)
            z_offset += local_Nz

        # Create GPU transport step for each GPU
        self.gpu_steps: List[GPUTransportStep] = []
        self.gpu_devices: List[int] = []

        for gpu_id in range(num_gpus):
            # Set GPU device
            with cp.cuda.Device(gpu_id):
                # Create transport step for local domain
                local_Nz = self.local_Nz_list[gpu_id]
                gpu_step = GPUTransportStep(
                    Ne=Ne,
                    Ntheta=Ntheta,
                    Nz=local_Nz + 2 * halo_depth,  # Include halo regions
                    Nx=Nx,
                    accumulation_mode=accumulation_mode,
                    delta_x=delta_x,
                    delta_z=delta_z,
                )
                self.gpu_steps.append(gpu_step)
                self.gpu_devices.append(gpu_id)

        print(f"Multi-GPU Transport Initialized:")
        print(f"  GPUs: {num_gpus}")
        print(f"  Total domain: {Ne}×{Ntheta}×{Nz}×{Nx}")
        print(f"  Decomposition: z-direction")
        for i, (local_Nz, offset) in enumerate(zip(self.local_Nz_list, self.z_offsets)):
            print(f"    GPU {i}: bins [{offset}:{offset+local_Nz}], size={local_Nz} + {2*halo_depth} halo")

        self.shape = (Ne, Ntheta, Nz, Nx)

    def _distribute_to_gpus(self, psi_global: cp.ndarray) -> List[cp.ndarray]:
        """Distribute global state to GPUs with halo regions.

        Args:
            psi_global: Global state [Ne, Ntheta, Nz, Nx]

        Returns:
            List of local states for each GPU with halos
        """
        psi_local_list = []

        for gpu_id, (local_Nz, z_offset) in enumerate(zip(self.local_Nz_list, self.z_offsets)):
            with cp.cuda.Device(gpu_id):
                # Extract local portion plus halos
                halo = self.halo_depth

                # Get z-range with halos (clamp to global domain)
                z_start_global = max(0, z_offset - halo)
                z_end_global = min(self.Nz_total, z_offset + local_Nz + halo)

                # Extract from global array
                psi_with_halo = psi_global[:, :, z_start_global:z_end_global, :].copy()

                # Pad if necessary (at boundaries)
                pad_top = halo - (z_offset - z_start_global)
                pad_bottom = halo - (z_end_global - (z_offset + local_Nz))

                if pad_top > 0 or pad_bottom > 0:
                    psi_padded = cp.zeros((self.Ne, self.Ntheta, local_Nz + 2*halo, self.Nx), dtype=cp.float32)
                    psi_padded[:, :, pad_top:pad_top+psi_with_halo.shape[2], :] = psi_with_halo
                    psi_local_list.append(psi_padded)
                else:
                    psi_local_list.append(psi_with_halo)

        return psi_local_list

    def _gather_from_gpus(self, psi_local_list: List[cp.ndarray]) -> cp.ndarray:
        """Gather local states from GPUs into global state.

        Args:
            psi_local_list: List of local states from each GPU

        Returns:
            Global state [Ne, Ntheta, Nz, Nx]
        """
        with cp.cuda.Device(0):  # Assemble on GPU 0
            psi_global = cp.zeros((self.Ne, self.Ntheta, self.Nz_total, self.Nx), dtype=cp.float32)

            for gpu_id, (local_Nz, z_offset) in enumerate(zip(self.local_Nz_list, self.z_offsets)):
                halo = self.halo_depth

                # Get local array from GPU
                psi_local = psi_local_list[gpu_id]

                # Extract non-halo region
                psi_non_halo = psi_local[:, :, halo:halo+local_Nz, :]

                # Copy to global array
                psi_global[:, :, z_offset:z_offset+local_Nz, :] = psi_non_halo

            return psi_global

    def _exchange_halos(self, psi_local_list: List[cp.ndarray]) -> None:
        """Exchange halo regions between adjacent GPUs.

        Args:
            psi_local_list: List of local states (modified in place)
        """
        halo = self.halo_depth

        for gpu_id in range(self.num_gpus):
            local_Nz = self.local_Nz_list[gpu_id]

            # Send to next GPU
            if gpu_id < self.num_gpus - 1:
                # Get data from current GPU's upper halo region
                with cp.cuda.Device(gpu_id):
                    send_data = psi_local_list[gpu_id][:, :, -2*halo:-halo, :].copy()

                # Put into next GPU's lower halo
                with cp.cuda.Device(gpu_id + 1):
                    psi_local_list[gpu_id + 1][:, :, :halo, :] = send_data

            # Receive from previous GPU
            if gpu_id > 0:
                # Get data from previous GPU's lower halo region
                with cp.cuda.Device(gpu_id - 1):
                    send_data = psi_local_list[gpu_id - 1][:, :, halo:2*halo, :].copy()

                # Put into current GPU's upper halo
                with cp.cuda.Device(gpu_id):
                    psi_local_list[gpu_id][:, :, -halo:, :] = send_data

    def apply_step(
        self,
        psi_global: cp.ndarray,
        E_grid: cp.ndarray,
        sigma_theta: float,
        theta_beam: float,
        delta_s: float,
        stopping_power,
        E_cutoff: float,
    ) -> Tuple[cp.ndarray, float, cp.ndarray]:
        """Apply multi-GPU transport step.

        Args:
            psi_global: Global state [Ne, Ntheta, Nz, Nx]
            E_grid: Energy grid [MeV]
            sigma_theta: RMS scattering angle
            theta_beam: Beam angle [rad]
            delta_s: Step length [mm]
            stopping_power: Stopping power [MeV/mm]
            E_cutoff: Cutoff energy [MeV]

        Returns:
            (psi_global_out, total_weight_leaked, total_deposited_energy) tuple
        """
        # Distribute to GPUs
        psi_local_list = self._distribute_to_gpus(psi_global)

        # Transport on each GPU
        psi_local_out_list = []
        total_weight_leaked = 0.0
        total_deposited = None

        for gpu_id in range(self.num_gpus):
            with cp.cuda.Device(gpu_id):
                psi_local = psi_local_list[gpu_id]
                gpu_step = self.gpu_steps[gpu_id]

                # Apply transport step
                psi_out, weight_leaked, deposited = gpu_step.apply_step(
                    psi=psi_local,
                    E_grid=E_grid,
                    sigma_theta=sigma_theta,
                    theta_beam=theta_beam,
                    delta_s=delta_s,
                    stopping_power=stopping_power,
                    E_cutoff=E_cutoff,
                )

                psi_local_out_list.append(psi_out)
                total_weight_leaked += weight_leaked

                # Accumulate deposited energy (on GPU 0)
                if total_deposited is None:
                    with cp.cuda.Device(0):
                        total_deposited = cp.zeros((self.Nz_total, self.Nx), dtype=cp.float32)

                # Add local contribution to global deposited energy
                local_Nz = self.local_Nz_list[gpu_id]
                z_offset = self.z_offsets[gpu_id]

                with cp.cuda.Device(0):
                    deposited_non_halo = deposited[:, halo:halo+local_Nz]
                    total_deposited[:, z_offset:z_offset+local_Nz] += deposited_non_halo

        # Exchange halos
        self._exchange_halos(psi_local_out_list)

        # Gather back to global
        psi_global_out = self._gather_from_gpus(psi_local_out_list)

        return psi_global_out, total_weight_leaked, total_deposited


def create_multi_gpu_transport_step(
    Ne: int,
    Ntheta: int,
    Nz: int,
    Nx: int,
    num_gpus: int = 4,
    accumulation_mode: str = AccumulationMode.FAST,
    delta_x: float = 1.0,
    delta_z: float = 1.0,
    halo_depth: int = 2,
) -> MultiGPUTransportStep:
    """Create multi-GPU transport step.

    Args:
        Ne: Number of energy bins
        Ntheta: Number of angular bins
        Nz: Number of depth bins
        Nx: Number of lateral bins
        num_gpus: Number of GPUs to use
        accumulation_mode: 'fast' or 'deterministic'
        delta_x: Lateral grid spacing [mm]
        delta_z: Depth grid spacing [mm]
        halo_depth: Number of halo bins for boundary exchange

    Returns:
        MultiGPUTransportStep instance
    """
    return MultiGPUTransportStep(
        Ne, Ntheta, Nz, Nx, num_gpus, accumulation_mode, delta_x, delta_z, halo_depth
    )
