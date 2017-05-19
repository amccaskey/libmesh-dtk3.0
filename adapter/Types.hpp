#ifndef DTK_TYPES_HPP
#define DTK_TYPES_HPP

#include <functional>
#include <limits>

namespace LibmeshAdapter
{
//! Entity id type.
typedef unsigned long int EntityId;

//! Invalid entity id.
static const EntityId dtk_invalid_entity_id =
    std::numeric_limits<EntityId>::max();

//! Support id type.
typedef unsigned long int SupportId;

//! Invalid support id.
static const SupportId dtk_invalid_support_id =
    std::numeric_limits<SupportId>::max();

// Forward declaration of Entity.
//class Entity;

//! Predicate function typedef.
//typedef std::function<bool( Entity )> PredicateFunction;

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_TYPES_HPP

//---------------------------------------------------------------------------//
// end DTK_Types.hpp
//---------------------------------------------------------------------------//
