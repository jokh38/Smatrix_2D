extern "C" __global__
void energy_loss_kernel_with_path_tracking(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    float* __restrict__ deposited_dose,
    double* __restrict__ escapes_gpu,
    const float* __restrict__ stopping_power_lut,
    const float* __restrict__ E_lut_grid,
    const float* __restrict__ E_phase_grid,
    const float* __restrict__ path_length_in,
    float* __restrict__ path_length_out,
    float delta_s,
    float E_cutoff,
    float E_initial,
    int Ne, int Ntheta, int Nz, int Nx,
    int lut_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    double local_energy_stopped = 0.0f;

    for (int idx = tid; idx < Ne * Ntheta * Nz * Nx; idx += total_threads) {
        int iE_in = idx / (Ntheta * Nz * Nx);
        int rem = idx % (Ntheta * Nz * Nx);
        int ith = rem / (Nz * Nx);
        rem = rem % (Nz * Nx);
        int iz = rem / Nx;
        int ix = rem % Nx;

        int src_idx = iE_in * E_stride + ith * theta_stride + iz * Nx + ix;
        float weight = psi_in[src_idx];

        if (weight < 1e-12f) {
            continue;
        }

        int spatial_idx = iz * Nx + ix;

        // For path tracking: read cumulative path length from previous step
        // This accumulates the actual distance traveled by particles at each (z,x) position
        float cumulative_path = path_length_in[spatial_idx];

        // Calculate effective energy based on cumulative path traveled
        // E_effective = E_initial - integral of stopping power along path
        // For 70 MeV protons in water: S varies from ~1.4 to ~12 MeV/mm
        // Use cumulative path directly from accumulated data
        float E_effective = E_initial - 1.72f * cumulative_path;

        // Update path length for output (accumulate for next step)
        atomicAdd(&path_length_out[spatial_idx], delta_s * weight);

        // If effective energy is below cutoff, absorb the particle
        if (E_effective <= E_cutoff) {
            atomicAdd(&deposited_dose[spatial_idx], weight * E_effective);
            local_energy_stopped += double(weight);
            continue;
        }

        // Get stopping power at the EFFECTIVE energy
        int lut_idx = 0;
        int left = 0;
        int right = lut_size - 2;
        while (left < right) {
            int mid = (left + right) / 2;
            if (E_effective < E_lut_grid[mid + 1]) {
                right = mid;
            } else {
                left = left + 1;
            }
        }
        lut_idx = left;

        float E0 = E_lut_grid[lut_idx];
        float E1 = E_lut_grid[min(lut_idx + 1, lut_size - 1)];
        float S0 = stopping_power_lut[lut_idx];
        float S1 = stopping_power_lut[min(lut_idx + 1, lut_size - 1)];
        float dE_lut = max(E1 - E0, 1e-12f);
        float frac = (E_effective - E0) / dE_lut;
        frac = fmaxf(0.0f, fminf(1.0f, frac));
        float S = S0 + frac * (S1 - S0);

        // Apply energy loss for this step
        float deltaE = S * delta_s;

        // Calculate new effective energy after this step
        float E_new = E_effective - deltaE;

        // Check if particle should be absorbed after this step
        if (E_new <= E_cutoff) {
            // Deposit remaining effective energy
            atomicAdd(&deposited_dose[spatial_idx], weight * E_effective);
            local_energy_stopped += double(weight);
            continue;
        }

        // Find output energy bin based on the NEW effective energy
        int iE_out = 0;
        if (E_new >= E_phase_grid[Ne - 1]) {
            iE_out = Ne - 1;
        } else if (E_new < E_phase_grid[0]) {
            atomicAdd(&deposited_dose[spatial_idx], weight * E_new);
            local_energy_stopped += double(weight);
            continue;
        } else {
            left = 0;
            right = Ne - 2;
            while (left < right) {
                int mid = (left + right) / 2;
                if (E_new < E_phase_grid[mid + 1]) {
                    right = mid;
                } else {
                    left = left + 1;
                }
            }
            iE_out = left;
        }

        if (iE_out >= Ne - 1) {
            int tgt_idx = (Ne - 1) * E_stride + ith * theta_stride + iz * Nx + ix;
            atomicAdd(&psi_out[tgt_idx], weight);
            atomicAdd(&deposited_dose[spatial_idx], weight * deltaE);
            continue;
        }

        float E_lo = E_phase_grid[iE_out];
        float E_hi = E_phase_grid[iE_out + 1];
        float dE_bin = max(E_hi - E_lo, 1e-12f);

        float w_lo = (E_hi - E_new) / dE_bin;
        float w_hi = 1.0f - w_lo;

        w_lo = fmaxf(0.0f, fminf(1.0f, w_lo));
        w_hi = fmaxf(0.0f, fminf(1.0f, w_hi));

        int tgt_lo = iE_out * E_stride + ith * theta_stride + iz * Nx + ix;
        int tgt_hi = (iE_out + 1) * E_stride + ith * theta_stride + iz * Nx + ix;

        atomicAdd(&psi_out[tgt_lo], weight * w_lo);
        atomicAdd(&psi_out[tgt_hi], weight * w_hi);
        atomicAdd(&deposited_dose[spatial_idx], weight * deltaE);
    }

    if (local_energy_stopped > 0.0) {
        atomicAdd(&escapes_gpu[2], local_energy_stopped);
    }
}
