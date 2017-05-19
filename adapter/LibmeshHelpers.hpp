#ifndef LIBMESHDTKADAPTERS_LIBMESHHELPERS_HPP
#define LIBMESHDTKADAPTERS_LIBMESHHELPERS_HPP

#include "LibmeshEntity.hpp"
#include "LibmeshEntityExtraData.hpp"

#include <Teuchos_Ptr.hpp>

#include <libmesh/mesh_base.h>

namespace LibmeshAdapter
{
//---------------------------------------------------------------------------//
/*!
  \class LibmeshHelpers
  \brief Libmesh helper functions
*/
//---------------------------------------------------------------------------//
class LibmeshHelpers
{
  public:
    // Extract the libmesh geom object.
    template <class LibmeshGeom>
    static Teuchos::Ptr<LibmeshGeom>
    extractGeom( const LibmeshEntity<LibmeshGeom> &entity )
    {
        return Teuchos::rcp_dynamic_cast<LibmeshEntityExtraData<LibmeshGeom>>(
                   entity.extraData() )
            ->d_libmesh_geom;
    }
};

//---------------------------------------------------------------------------//

} // end namespace LibmeshAdapter

#endif // end LIBMESHDTKADAPTERS_LIBMESHHELPERS_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshHelpers.hpp
//---------------------------------------------------------------------------//
