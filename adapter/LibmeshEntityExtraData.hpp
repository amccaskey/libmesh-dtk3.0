#ifndef LIBMESHDTKADAPTERS_LIBMESHENTITYEXTRADATA_HPP
#define LIBMESHDTKADAPTERS_LIBMESHENTITYEXTRADATA_HPP

#include <Teuchos_Ptr.hpp>

#include <libmesh/elem.h>

namespace LibmeshAdapter
{
//---------------------------------------------------------------------------//
/*!
  \class LibmeshEntityExtraData
  \brief A base class for setting extra data with entities.
*/
//---------------------------------------------------------------------------//
template <class LibmeshGeom>
class LibmeshEntityExtraData {
  public:
    LibmeshEntityExtraData( const Teuchos::Ptr<LibmeshGeom> &libmesh_geom )
        : d_libmesh_geom( libmesh_geom )
    { /* ... */
    }

    ~LibmeshEntityExtraData() { /* ... */}

    // libMesh geom.
    Teuchos::Ptr<LibmeshGeom> d_libmesh_geom;
};

//---------------------------------------------------------------------------//

} // end namespace LibmeshAdapter

#endif // end LIBMESHDTKADAPTERS_LIBMESHENTITYEXTRADATA_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshEntityExtraData.hpp
//---------------------------------------------------------------------------//
