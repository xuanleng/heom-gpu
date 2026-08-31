#ifndef RK4_INTEGRATOR_CUH
#define RK4_INTEGRATOR_CUH

#include "heom_derivatives.cuh"

/**
 * @brief RK4 time integrator for HEOM
 *
 * Implements 4th-order Runge-Kutta:
 *   k1 = f(t, y)
 *   k2 = f(t + dt/2, y + dt/2 * k1)
 *   k3 = f(t + dt/2, y + dt/2 * k2)
 *   k4 = f(t + dt, y + dt * k3)
 *   y_next = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 *
 * The combine is IN-PLACE: on return, rho_in's storage holds y_next (the
 * buffer is mutable HEOMTrajectory memory; rho_in is const only as the
 * derivative kernels' read-contract). ctx.drho_out is ignored — the stage
 * epilogues are FUSED into the derivative kernel, so each stage's k lives
 * only in registers (4 launches per step, no k buffer, no swap).
 *
 * @param ctx  HEOM context (rho_in = current state, updated in-place)
 * @param dt   Time step (a.u.)
 * @param workspace  Caller-allocated buffer [3 × batch × M × N × N] complex_t
 *                   (temp_a + temp_b + acc, each spanning all batch slots; see
 *                   rk4_step() for the layout). batch==1 ⇒ 3 × M × N × N.
 * @param kahan_comp Persistent [batch × M × N × N] complex_t compensation buffer for
 *                   the final combine, or nullptr. nullptr ⇒ plain combine
 *                   (FP64 build stays bit-identical to history). Non-null ⇒
 *                   Kahan-compensated accumulation into ρ, the
 *                   mixed-precision path's defence against T·eps drift. The
 *                   buffer MUST persist across steps (the carry lives in it;
 *                   zeroing it mid-run defeats the point) — HEOMTrajectory owns
 *                   and zero-inits it once.
 */
void rk4_step(const HEOMContext& ctx, double dt, heom::complex_t* workspace,
              heom::complex_t* kahan_comp = nullptr);

/**
 * @brief Integrating-factor RK4 step (Lawson): Term 2 (hierarchy decay g_α) is
 *        integrated EXACTLY per-ADO via the precomputed factors, RK4 handles
 *        only the non-stiff coupling. Same 4-launch structure, workspace layout
 *        and in-place semantics as rk4_step (reuses [temp_a|temp_b|acc]).
 *
 * @param e_half device [M] exp(-½·dt·g_α)  (must match the dt passed here)
 * @param e_full device [M] exp(-dt·g_α)
 *
 * The factors are a function of dt; the caller (HEOMTrajectory) recomputes them
 * via compute_if_factors only when dt changes. Other params as rk4_step.
 */
void rk4_step_if(const HEOMContext& ctx, double dt, heom::complex_t* workspace,
                 heom::complex_t* kahan_comp,
                 const heom::real_t* e_half, const heom::real_t* e_full);


#endif // RK4_INTEGRATOR_CUH
