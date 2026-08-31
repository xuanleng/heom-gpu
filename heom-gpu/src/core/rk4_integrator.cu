#include "rk4_integrator.cuh"
#include "heom_derivatives.cuh"
#include "cuda_common.cuh"
#include "core/precision.hpp"
#include <cuComplex.h>

using heom::complex_t;
using heom::c_make;

// 3-buffer RK4 with the stage epilogue FUSED into the derivative kernel
// (heom_derivatives.cu, RK4Fuse): each stage's k is consumed straight from
// the derivative kernel's register — never stored to or reloaded from a
// k-buffer. Per step: 4 launches total (was 8), and the M·N² k round-trip
// (one store + one load per stage) is gone. The derivative kernel is
// compute-bound, so the fused element-wise epilogue hides under it.
//
// Workspace = [temp_a | temp_b | acc], 3 · M · N² · sizeof(complex_t):
//   - temp_a/temp_b : ping-pong f() inputs. The derivative reads its input
//     CROSS-element (Term 1 row/column, Term 4/5 neighbors), so a stage must
//     never write the buffer it is reading — hence two temps, alternating
//     rho → a → b → a. (Same total footprint as the old k|temp|acc split.)
//   - acc           : running sum k1 + 2·k2 + 2·k3 (k4 folded in the final).
//
// The final combine is in-place into rho on both paths (no second rho buffer,
// no swap). Whole-step (u_list set) shrinks every launch from M·N² to |U|·N²
// and freezes α∉U by never indexing it.
void rk4_step(const HEOMContext& ctx, double dt, complex_t* workspace,
              complex_t* kahan_comp)
{
    HEOMContext step_ctx = ctx;

    complex_t* rho = const_cast<complex_t*>(step_ctx.rho_in);
    // per_slot = M·N² keeps the int overflow guard; the workspace is laid out as
    // [temp_a | temp_b | acc], each sub-buffer spanning ALL batch slots
    // (batch·M·N²). The derivative kernel writes slot b at boff = b·M·N² inside
    // whichever sub-buffer it is handed, so the three bases must be batch·M·N²
    // apart. size_t stride: batch·M·N² can exceed the int per-slot ceiling.
    const int per_slot = heom::checked_total_elements(step_ctx.M, step_ctx.N);
    const size_t stride = static_cast<size_t>(step_ctx.batch) * per_slot;

    complex_t* temp_a = workspace;
    complex_t* temp_b = workspace + stride;
    complex_t* acc    = workspace + 2 * stride;

    // RK4 stage scalars are exact small integers/halves — representable in FP32
    // without loss, so c_make (float in mixed) introduces no error here.
    const complex_t half_dt = c_make(dt / 2.0, 0.0);
    const complex_t full_dt = c_make(dt,       0.0);
    const complex_t one     = c_make(1.0,      0.0);
    const complex_t two     = c_make(2.0,      0.0);
    const complex_t dt_6th  = c_make(dt / 6.0, 0.0);

    heom::RK4StageArgs st;
    st.rho = rho;
    st.acc = acc;

    // k1 = f(rho);     temp_a = rho + (dt/2)·k1,  acc = k1
    st.temp_out = temp_a;  st.c = half_dt;  st.init_acc = true;
    compute_heom_derivatives_stage(step_ctx, st);
    st.init_acc = false;

    // k2 = f(temp_a);  temp_b = rho + (dt/2)·k2,  acc += 2·k2
    step_ctx.rho_in = temp_a;
    st.temp_out = temp_b;  st.c = half_dt;  st.w = two;
    compute_heom_derivatives_stage(step_ctx, st);

    // k3 = f(temp_b);  temp_a = rho + dt·k3,      acc += 2·k3
    step_ctx.rho_in = temp_b;
    st.temp_out = temp_a;  st.c = full_dt;  st.w = two;
    compute_heom_derivatives_stage(step_ctx, st);

    // k4 = f(temp_a);  rho += (dt/6)·(k4 + acc)   [in-place, opt. Kahan]
    step_ctx.rho_in = temp_a;
    st.comp = kahan_comp;  st.w = one;  st.dt6 = dt_6th;
    compute_heom_derivatives_final(step_ctx, st);

    // Each fused launch validates + checks itself inside
    // compute_heom_derivatives_*; nothing further to verify here.
}

// Integrating-factor RK4 (Lawson): identical 4-launch skeleton to rk4_step, but
// Term 2 (g_α) is folded EXACTLY through the per-ADO E/E2 factors in the fused
// epilogue (see heom_derivatives.cu). The RK4 derivative excludes Term 2, so
// the step never sees the stiff decay eigenvalue → escapes the Term-2 dt wall.
// Workspace layout, ping-pong and in-place final are byte-for-byte the same as
// rk4_step; only the entry points (_stage_if/_final_if) and st.e_half/e_full/
// if_stage differ.
void rk4_step_if(const HEOMContext& ctx, double dt, complex_t* workspace,
                 complex_t* kahan_comp,
                 const heom::real_t* e_half, const heom::real_t* e_full)
{
    HEOMContext step_ctx = ctx;

    complex_t* rho = const_cast<complex_t*>(step_ctx.rho_in);
    const int per_slot = heom::checked_total_elements(step_ctx.M, step_ctx.N);
    const size_t stride = static_cast<size_t>(step_ctx.batch) * per_slot;

    complex_t* temp_a = workspace;
    complex_t* temp_b = workspace + stride;
    complex_t* acc    = workspace + 2 * stride;

    const complex_t half_dt = c_make(dt / 2.0, 0.0);
    const complex_t full_dt = c_make(dt,       0.0);
    const complex_t one     = c_make(1.0,      0.0);
    const complex_t two     = c_make(2.0,      0.0);
    const complex_t dt_6th  = c_make(dt / 6.0, 0.0);

    heom::RK4StageArgs st;
    st.rho    = rho;
    st.acc    = acc;
    st.e_half = e_half;
    st.e_full = e_full;

    // k1 = F(rho);   temp_a = E·rho + E·(dt/2)·k1,  acc = E2·k1
    st.temp_out = temp_a;  st.c = half_dt;  st.if_stage = 1;
    compute_heom_derivatives_stage_if(step_ctx, st);

    // k2 = F(temp_a); temp_b = E·rho + (dt/2)·k2,   acc += 2·E·k2
    step_ctx.rho_in = temp_a;
    st.temp_out = temp_b;  st.c = half_dt;  st.w = two;  st.if_stage = 2;
    compute_heom_derivatives_stage_if(step_ctx, st);

    // k3 = F(temp_b); temp_a = E2·rho + E·dt·k3,    acc += 2·E·k3
    step_ctx.rho_in = temp_b;
    st.temp_out = temp_a;  st.c = full_dt;  st.w = two;  st.if_stage = 3;
    compute_heom_derivatives_stage_if(step_ctx, st);

    // k4 = F(temp_a); rho = E2·rho + (dt/6)·(k4 + acc)  [in-place, opt. Kahan]
    step_ctx.rho_in = temp_a;
    st.comp = kahan_comp;  st.w = one;  st.dt6 = dt_6th;
    compute_heom_derivatives_final_if(step_ctx, st);
}
