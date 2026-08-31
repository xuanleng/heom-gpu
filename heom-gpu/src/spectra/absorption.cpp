#include "spectra/absorption.hpp"
#include "spectra/linear_response.hpp"

namespace heom::spectra {

LinearResponseResult compute_absorption(
    const SystemContext& sys,
    const DipoleOperator& mu)
{
    // LinearResponseRequest carries only the dipole: compute_linear_response
    // hard-codes the absorption algorithm because there is exactly one sensible
    // choice. Future spectra get their own compute_* functions.
    LinearResponseRequest req;
    req.dipole = mu;
    return compute_linear_response(sys, req);
}

}  // namespace heom::spectra
