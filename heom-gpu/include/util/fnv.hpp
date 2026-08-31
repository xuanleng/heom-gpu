#ifndef HEOM_UTIL_FNV_HPP
#define HEOM_UTIL_FNV_HPP

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace heom {

// FNV-1a 64-bit of `text`, as 16 zero-padded hex chars. Single source of truth
// for content fingerprints: config_hash (heom_config) and the MC merge key
// (mc_driver) both call this, so a shard's key is reproducible from its inputs.
inline std::string fnv1a64_hex(const std::string& text) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

} // namespace heom

#endif // HEOM_UTIL_FNV_HPP
