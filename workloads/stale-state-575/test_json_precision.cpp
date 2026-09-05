// SPDX-License-Identifier: MIT
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

int main()
{
    constexpr unsigned long long start_ns = 162772662173223ULL;
    constexpr unsigned long long end_ns = 162774693040424ULL;
    const double expected_ms = static_cast<double>(end_ns - start_ns) / 1.0e6;
    std::ostringstream output;

    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << expected_ms;
    const std::string encoded = output.str();
    char *end = nullptr;
    const double parsed_ms = std::strtod(encoded.c_str(), &end);
    if (end == encoded.c_str() || *end != '\0' || parsed_ms != expected_ms ||
        encoded == "2030.87") {
        std::cerr << "JSON floating-point precision did not round trip\n";
        return 1;
    }
    std::cout << "json_precision: non-round nanosecond duration round trips\n";
    return 0;
}
