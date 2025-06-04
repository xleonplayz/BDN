# NV-Simulator extracted from Untitled16.ipynb
# Ported to standalone Python script

import jax
import jax.numpy as jnp
from jax import random, jit, lax
from dataclasses import dataclass
from typing import Tuple


###############################################################################
#  NV-ZENTRUM MONTE-CARLO with ¹⁴N hyperfine interaction
###############################################################################

@dataclass(frozen=True)
class NVParamsJAX:
    """Physical parameters (units: ns, ns⁻¹)"""
    gamma_rad: float      = 1.0 / 12.0       # Radiative decay (ns⁻¹)
    k_ISC_ms0: float      = 1.0 / 300.0      # ISC ms=0 (ns⁻¹)
    k_ISC_ms1: float      = 1.0 / 25.0       # ISC ms=±1 (ns⁻¹)
    gamma_S: float        = 1.0 / 220.0      # Singlet decay (ns⁻¹)
    p_S_to_ms0: float     = 0.90             # Singlet → ms=0
    p_S_to_ms1: float     = 0.10             # Singlet → ms=±1
    gamma_pump: float     = 0.01             # Laser pump (ns⁻¹)
    gamma_orb: float      = 0.1              # Jahn-Teller (ns⁻¹)
    A_par_ns: float       = 2.16e6 / 1e9     # Hyperfine ¹⁴N (ns⁻¹)
    omega_rabi_Hz: float  = 5.0e6            # Rabi frequency (Hz)
    mw_freq_Hz: float     = 2.87e9           # MW frequency (Hz)
    detection_eff: float  = 1.0              # Detection efficiency
    dark_count_rate: float= 0.0005           # Darkcount rate (ns⁻¹)
    delta_e_ns: float     = 30e6 / 1e9       # Orbital splitting (ns⁻¹)


# ---------------------------------------------------------------------------
#  Helper: index conversion & hyperfine shift
# ---------------------------------------------------------------------------

@jit
def idx_to_tuple(idx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    elec = idx // 3
    mI   = (idx % 3) - 1
    return elec, mI

@jit
def tuple_to_idx(elec: jnp.ndarray, mI: jnp.ndarray) -> jnp.ndarray:
    return elec * 3 + (mI + 1)

@jit
def hyperfine_shift(elec: jnp.ndarray, mI: jnp.ndarray, params: NVParamsJAX) -> jnp.ndarray:
    return jnp.where(elec == 1, params.A_par_ns * mI, 0.0)


# ---------------------------------------------------------------------------
#  Transition rate calculation (vectorised)
# ---------------------------------------------------------------------------

@jit
def get_transition_rates(state: jnp.ndarray,
                         laser_on: bool,
                         params: NVParamsJAX) -> Tuple[jnp.ndarray, jnp.ndarray]:
    batch = state.shape[0]
    max_t = 4
    targets = jnp.zeros((batch, max_t), dtype=jnp.int32)
    rates   = jnp.zeros((batch, max_t))

    elec, mI = idx_to_tuple(state)

    def per_run(e, mi, s):
        t = jnp.zeros((max_t,), dtype=jnp.int32)
        r = jnp.zeros((max_t,))

        def case_ground0(args):
            t, r = args
            t = t.at[0].set(tuple_to_idx(2, mi))
            r = r.at[0].set(0.5 * params.gamma_pump)
            t = t.at[1].set(tuple_to_idx(3, mi))
            r = r.at[1].set(0.5 * params.gamma_pump)
            return t, r

        def case_ground1(args):
            t, r = args
            t = t.at[0].set(tuple_to_idx(4, mi))
            r = r.at[0].set(0.5 * params.gamma_pump)
            t = t.at[1].set(tuple_to_idx(5, mi))
            r = r.at[1].set(0.5 * params.gamma_pump)
            return t, r

        def case_Ex_ms0(args):
            t, r = args
            t = t.at[0].set(tuple_to_idx(0, mi))
            r = r.at[0].set(params.gamma_rad)
            t = t.at[1].set(tuple_to_idx(6, mi))
            r = r.at[1].set(params.k_ISC_ms0)
            t = t.at[2].set(tuple_to_idx(3, mi))
            r = r.at[2].set(params.gamma_orb)
            return t, r

        def case_Ey_ms0(args):
            t, r = args
            t = t.at[0].set(tuple_to_idx(0, mi))
            r = r.at[0].set(params.gamma_rad)
            t = t.at[1].set(tuple_to_idx(6, mi))
            r = r.at[1].set(params.k_ISC_ms0)
            t = t.at[2].set(tuple_to_idx(2, mi))
            r = r.at[2].set(params.gamma_orb)
            return t, r

        def case_Ex_ms1(args):
            t, r = args
            t = t.at[0].set(tuple_to_idx(1, mi))
            r = r.at[0].set(params.gamma_rad)
            t = t.at[1].set(tuple_to_idx(6, mi))
            r = r.at[1].set(params.k_ISC_ms1)
            t = t.at[2].set(tuple_to_idx(5, mi))
            r = r.at[2].set(params.gamma_orb)
            return t, r

        def case_Ey_ms1(args):
            t, r = args
            t = t.at[0].set(tuple_to_idx(1, mi))
            r = r.at[0].set(params.gamma_rad)
            t = t.at[1].set(tuple_to_idx(6, mi))
            r = r.at[1].set(params.k_ISC_ms1)
            t = t.at[2].set(tuple_to_idx(4, mi))
            r = r.at[2].set(params.gamma_orb)
            return t, r

        def case_S(args):
            t, r = args
            t = t.at[0].set(tuple_to_idx(0, mi))
            r = r.at[0].set(params.gamma_S * params.p_S_to_ms0)
            t = t.at[1].set(tuple_to_idx(1, mi))
            r = r.at[1].set(params.gamma_S * params.p_S_to_ms1)
            return t, r

        t, r = lax.switch(e,
                          branch=(case_ground0, case_ground1,
                                  case_Ex_ms0, case_Ey_ms0,
                                  case_Ex_ms1, case_Ey_ms1,
                                  case_S),
                          operand=(t, r))
        return t, r

    targets, rates = jax.vmap(per_run)(elec, mI, state)
    return targets, rates


# ---------------------------------------------------------------------------
#  PI-PULSE under hyperfine detuning (vector)
# ---------------------------------------------------------------------------

@jit
def apply_pi_pulse(state: jnp.ndarray,
                   pulse_duration: float,
                   params: NVParamsJAX) -> jnp.ndarray:
    elec, mI = idx_to_tuple(state)
    D_ns = 2.87e9 / 1e9
    ω_MW_ns = params.mw_freq_Hz / 1e9
    Δ_hf = jnp.where(elec == 1, params.A_par_ns * mI, 0.0)
    ω_res = D_ns + Δ_hf
    Ω_ns = 2 * jnp.pi * (params.omega_rabi_Hz / 1e9)
    t_pi = jnp.pi / Ω_ns
    tol  = 2.0

    on_ground = (elec == 0) | (elec == 1)
    in_res    = jnp.abs(ω_res - ω_MW_ns) < (Ω_ns / 2.0)
    correct_t = jnp.abs(pulse_duration - t_pi) < tol

    do_flip = on_ground & in_res & correct_t

    new_elec = jnp.where((elec == 0) & do_flip, 1,
                         jnp.where((elec == 1) & do_flip, 0, elec))
    return tuple_to_idx(new_elec, mI)


# ---------------------------------------------------------------------------
#  Monte-Carlo steps (JAX while-loop)
# ---------------------------------------------------------------------------

def simulate_one_run(rng_key: jnp.ndarray,
                     params: NVParamsJAX,
                     polarize_ns: float,
                     mw_pulse_ns: float,
                     wait_ns: float,
                     read_ns: float,
                     max_photons: int,
                     max_steps: int) -> jnp.ndarray:
    state = jnp.array(1, dtype=jnp.int32)  # |g, ms=0, mI=0>
    time  = 0.0
    photon_times = jnp.full((max_photons,), read_ns + 1.0)
    photon_idx   = 0
    key = rng_key

    # -- 1) Pre-pump --
    def cond_pre(carry):
        t, _, _, _, _ = carry
        return t < polarize_ns

    def body_pre(carry):
        t, st, ph, pi, k = carry
        targets, rates = get_transition_rates(st, True, params)
        total_rate = jnp.sum(rates)
        k, sub = random.split(k)
        dt = random.exponential(sub) / total_rate
        t_next = t + dt
        do_jump = t_next <= polarize_ns
        t_new = jnp.where(do_jump, t_next, polarize_ns)

        k, sub2 = random.split(k)
        r = random.uniform(sub2) * total_rate
        cumsum = jnp.cumsum(rates)
        idx_jump = jnp.argmin(jnp.where(cumsum > r, cumsum, total_rate + 1.0))
        st_new = jnp.where(do_jump, targets[idx_jump], st)
        return (t_new, st_new, ph, pi, k)

    carry = (time, state, photon_times, photon_idx, key)
    carry = lax.while_loop(cond_pre, body_pre, carry)
    time, state, photon_times, photon_idx, key = carry

    # -- 2) π-pulse --
    state = apply_pi_pulse(state, mw_pulse_ns, params)
    time = time + mw_pulse_ns

    # -- 3) Wait time --
    time = time + wait_ns
    read_start = time
    read_end   = time + read_ns

    # -- 4) Readout --
    def cond_read(carry):
        t, _, _, pi, _, steps = carry
        return (t < read_end) & (pi < max_photons) & (steps < max_steps)

    def body_read(carry):
        t, st, pt, pi, k, steps = carry
        targets, rates = get_transition_rates(st, True, params)
        total_rate = jnp.sum(rates)
        k, sub = random.split(k)
        dt = random.exponential(sub) / total_rate
        t_next = t + dt
        do_jump = t_next <= read_end
        t_new = jnp.where(do_jump, t_next, read_end)

        k, sub2 = random.split(k)
        r = random.uniform(sub2) * total_rate
        cumsum = jnp.cumsum(rates)
        idx_jump = jnp.argmin(jnp.where(cumsum > r, cumsum, total_rate + 1.0))
        st_new = jnp.where(do_jump, targets[idx_jump], st)

        old_elec, _ = idx_to_tuple(st)
        new_elec, _ = idx_to_tuple(st_new)
        is_photon = ((old_elec < 6) & (new_elec < 2) &
                     (((old_elec == 2) & (new_elec == 0)) |
                      ((old_elec == 3) & (new_elec == 0)) |
                      ((old_elec == 4) & (new_elec == 1)) |
                      ((old_elec == 5) & (new_elec == 1))))
        k, sub3 = random.split(k)
        detect = random.uniform(sub3) < params.detection_eff

        t_rel = t_new - read_start
        pt = pt.at[pi].set(jnp.where(do_jump & is_photon & detect, t_rel, pt[pi]))
        pi = pi + jnp.where(do_jump & is_photon & detect, 1, 0)

        return (t_new, st_new, pt, pi, k, steps + 1)

    carry = (time, state, photon_times, photon_idx, key, 0)
    carry = lax.while_loop(cond_read, body_read, carry)
    _, _, photon_times, _, _, _ = carry

    return photon_times

# JIT compile
simulate_one_run = jit(simulate_one_run, static_argnums=(1,))


# ---------------------------------------------------------------------------
#  Wrapper for multiple runs using vmap
# ---------------------------------------------------------------------------

def simulate_n_runs(rng_key: jnp.ndarray,
                    params: NVParamsJAX,
                    n_runs: int,
                    polarize_ns: float,
                    mw_pulse_ns: float,
                    wait_ns: float,
                    read_ns: float,
                    max_photons: int,
                    max_steps: int) -> jnp.ndarray:
    keys = random.split(rng_key, num=n_runs)
    results = jax.vmap(lambda k: simulate_one_run(k, params,
                                                  polarize_ns,
                                                  mw_pulse_ns,
                                                  wait_ns,
                                                  read_ns,
                                                  max_photons,
                                                  max_steps))(keys)
    return results

# JIT compile
simulate_n_runs = jit(simulate_n_runs, static_argnums=(1, 2))


# ---------------------------------------------------------------------------
#  Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as onp
    import matplotlib.pyplot as plt

    seed = 42
    rng = random.PRNGKey(seed)

    params = NVParamsJAX(
        gamma_rad       = 1.0 / 12.0,
        k_ISC_ms0       = 1.0 / 300.0,
        k_ISC_ms1       = 1.0 / 25.0,
        gamma_S         = 1.0 / 220.0,
        p_S_to_ms0      = 0.90,
        p_S_to_ms1      = 0.10,
        gamma_pump      = 0.01,
        gamma_orb       = 0.1,
        A_par_ns        = 2.16e6 / 1e9,
        omega_rabi_Hz   = 5e6,
        mw_freq_Hz      = 2.87e9,
        detection_eff   = 1.0,
        dark_count_rate = 0.0005,
        delta_e_ns      = 30e6 / 1e9
    )

    # Pulse parameters (ns)
    POLARIZE_NS = 300.0
    MW_PULSE_NS = 100.0
    WAIT_NS     = 20.0
    READ_NS     = 3000.0

    N_RUNS      = 1000       # number of trajectories
    MAX_PHOTONS = 100        # padded photon list
    MAX_STEPS   = 5000       # max jumps per run

    photon_times_array = simulate_n_runs(rng, params,
                                         N_RUNS,
                                         POLARIZE_NS,
                                         MW_PULSE_NS,
                                         WAIT_NS,
                                         READ_NS,
                                         MAX_PHOTONS,
                                         MAX_STEPS)

    pt_np = onp.array(photon_times_array)
    valid_times = pt_np[pt_np <= READ_NS]

    BIN_WIDTH = 10.0
    bins = onp.arange(0, READ_NS + BIN_WIDTH, BIN_WIDTH)
    counts_per_bin = onp.histogram(valid_times, bins=bins)[0]

    dark = onp.random.poisson(params.dark_count_rate * BIN_WIDTH,
                              size=counts_per_bin.shape)
    counts_per_bin = counts_per_bin + dark

    centers = bins[:-1] + BIN_WIDTH / 2.0
    plt.figure(figsize=(10, 6))
    plt.bar(centers, counts_per_bin, width=BIN_WIDTH,
            color='green', edgecolor='darkgreen', alpha=0.7)
    plt.xlabel('Zeit seit Readout-Start (ns)')
    plt.ylabel('Counts / 10 ns-Bin (über alle Runs)')
    plt.title(f'NV-Zentrum Monte Carlo ({N_RUNS} Läufe)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pulse_histogram.png")
    print("Saved pulse histogram to pulse_histogram.png")
