# Base NV center simulation functions extracted from Untitled16.ipynb
import jax
import jax.numpy as jnp
from jax import random, jit, lax
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class NVParams:
    gamma_rad: float = 1.0 / 12.0
    k_ISC_ms0: float = 1.0 / 300.0
    k_ISC_ms1: float = 1.0 / 25.0
    gamma_S: float = 1.0 / 220.0
    p_S_to_ms0: float = 0.90
    p_S_to_ms1: float = 0.10
    gamma_pump: float = 0.01
    gamma_orb: float = 0.1
    A_par_ns: float = 2.16e6 / 1e9
    omega_rabi_Hz: float = 5e6
    mw_freq_Hz: float = 2.87e9
    detection_eff: float = 1.0
    dark_count_rate: float = 0.0005
    delta_e_ns: float = 30e6 / 1e9

@jit
def idx_to_tuple(idx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    elec = idx // 3
    mI = (idx % 3) - 1
    return elec, mI

@jit
def tuple_to_idx(elec: jnp.ndarray, mI: jnp.ndarray) -> jnp.ndarray:
    return elec * 3 + (mI + 1)

@jit
def get_transition_rates(state: jnp.ndarray, laser_on: bool, params: NVParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
    batch = state.shape[0]
    max_t = 4
    targets = jnp.zeros((batch, max_t), dtype=jnp.int32)
    rates = jnp.zeros((batch, max_t))

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

        t, r = lax.switch(e, branch=(case_ground0, case_ground1,
                                     case_Ex_ms0, case_Ey_ms0,
                                     case_Ex_ms1, case_Ey_ms1,
                                     case_S), operand=(t, r))
        return t, r

    targets, rates = jax.vmap(per_run)(elec, mI, state)
    return targets, rates

@jit
def apply_pi_pulse(state: jnp.ndarray, pulse_duration: float, params: NVParams) -> jnp.ndarray:
    elec, mI = idx_to_tuple(state)
    D_ns = 2.87e9 / 1e9
    omega_MW_ns = params.mw_freq_Hz / 1e9
    delta_hf = jnp.where(elec == 1, params.A_par_ns * mI, 0.0)
    omega_res = D_ns + delta_hf
    Omega_ns = 2 * jnp.pi * (params.omega_rabi_Hz / 1e9)
    t_pi = jnp.pi / Omega_ns
    tol = 2.0

    on_ground = (elec == 0) | (elec == 1)
    in_res = jnp.abs(omega_res - omega_MW_ns) < (Omega_ns / 2.0)
    correct_t = jnp.abs(pulse_duration - t_pi) < tol

    do_flip = on_ground & in_res & correct_t

    new_elec = jnp.where((elec == 0) & do_flip, 1,
                         jnp.where((elec == 1) & do_flip, 0, elec))
    return tuple_to_idx(new_elec, mI)


def simulate_one_run(rng_key: jnp.ndarray,
                     params: NVParams,
                     polarize_ns: float,
                     mw_pulse_ns: float,
                     wait_ns: float,
                     read_ns: float,
                     max_photons: int,
                     max_steps: int) -> jnp.ndarray:
    state = jnp.array(1, dtype=jnp.int32)
    time = 0.0
    photon_times = jnp.full((max_photons,), read_ns + 1.0)
    photon_idx = 0
    key = rng_key

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

    state = apply_pi_pulse(state, mw_pulse_ns, params)
    time = time + mw_pulse_ns
    time = time + wait_ns
    read_start = time
    read_end = time + read_ns

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

simulate_one_run = jit(simulate_one_run, static_argnums=(1,))


def simulate_n_runs(rng_key: jnp.ndarray,
                    params: NVParams,
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

simulate_n_runs = jit(simulate_n_runs, static_argnums=(1, 2))
