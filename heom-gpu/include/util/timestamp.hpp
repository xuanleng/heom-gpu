#ifndef HEOM_UTIL_TIMESTAMP_HPP
#define HEOM_UTIL_TIMESTAMP_HPP

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace heom {

// Local wall-clock as `YYYYMMDD_HHMMSS`. The one stem used for every
// timestamped artefact: the `use_timestamp: true` output file name
// (resolve_output_path) and the MC worker's default --run-id, so a shard with
// no explicit run-id is named exactly like a single-shot run would be. Second
// resolution: an auto-named file is created exclusively (create_output_file),
// so two runs of one stem collide only if they write within the same second.
inline std::string local_timestamp() {
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream ts;
    ts << std::setfill('0')
       << std::setw(4) << (tm.tm_year + 1900)
       << std::setw(2) << (tm.tm_mon + 1)
       << std::setw(2) << tm.tm_mday << "_"
       << std::setw(2) << tm.tm_hour
       << std::setw(2) << tm.tm_min
       << std::setw(2) << tm.tm_sec;
    return ts.str();
}

} // namespace heom

#endif // HEOM_UTIL_TIMESTAMP_HPP
