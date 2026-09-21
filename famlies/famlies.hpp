#ifndef FAMLIES_FAMLIES_HPP
#define FAMLIES_FAMLIES_HPP

#include <blis.h>

namespace famlies
{

/// The version string of the BLIS build famlies is linked against.
inline const char* blis_version()
{
    return bli_info_get_version_str();
}

} // namespace famlies

#endif // FAMLIES_FAMLIES_HPP
