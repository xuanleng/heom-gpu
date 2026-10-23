#ifndef HEOM_MODE_LAYOUT_HPP
#define HEOM_MODE_LAYOUT_HPP

// (alpha, n, k) -> flat index into every per-ADO mode table:
//   level_indices, neighbor_map_plus/minus, coeff_plus/minus  [M x D]
//   c_bath, nu_bath                                           [D]
//
// Bath n owns the contiguous slot range [off[n], off[n+1]); D = off[n_bath] is
// the total ladder count N_ell = Σ_n P_n. Per-bath slot counts may differ
// (a pigment with vibrational modes carries extra ladders). The uniform
// case off[n] = n·(K+1) reproduces the old rectangular layout number for
// number, so nothing downstream can tell the two apart.
//
// ONE definition, shared by the host precompute (system_context), the device
// Term 4/5 loop (heom_derivatives), the filter statistics (ado_filter), the
// hierarchy neighbor maps (hierarchy_truncation), the u-list channel count
// (ulist_builder) and the allocation plan (gpu_alloc_plan).
//
// Non-owning view: `off` points at PhysicalParameters::mode_off on the host
// and at SystemContext::d_mode_off_ on the device. Header-only and
// CUDA-agnostic on purpose: host .cpp files include it without pulling in
// cuda_runtime.

#ifdef __CUDACC__
#define HEOM_HD __host__ __device__
#else
#define HEOM_HD
#endif

namespace heom {

struct ModeLayout {
    int        n_bath;   // sites carrying a bath (== n_states, not the Hilbert dim)
    int        D;        // total slots per ADO, == off[n_bath]
    const int* off;      // [n_bath+1] prefix sums, off[0] == 0

    HEOM_HD int channels() const           { return D; }                 // per ADO
    HEOM_HD int count(int n) const         { return off[n + 1] - off[n]; }
    HEOM_HD int slot(int n, int k) const   { return off[n] + k; }        // within ADO
    HEOM_HD int idx(int alpha, int n, int k) const {
        return alpha * D + off[n] + k;
    }
    // Inverse of slot(). Linear scan over n_bath+1 ints: only used on error
    // paths and for metadata, never in a kernel hot loop.
    HEOM_HD int site_of(int s) const {
        int n = 0;
        while (n + 1 < n_bath && off[n + 1] <= s) ++n;
        return n;
    }
    HEOM_HD int pole_of(int s) const       { return s - off[site_of(s)]; }
};

}  // namespace heom

#endif  // HEOM_MODE_LAYOUT_HPP
