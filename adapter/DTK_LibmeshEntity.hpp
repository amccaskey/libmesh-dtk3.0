#ifndef LIBMESHDTKADAPTERS_LIBMESHENTITY_HPP
#define LIBMESHDTKADAPTERS_LIBMESHENTITY_HPP

#include "DTK_LibmeshAdjacencies.hpp"
#include <DTK_Types.hpp>
#include <Teuchos_Ptr.hpp>
#include <libmesh/mesh_base.h>

#include "DTK_LibmeshEntityExtraData.hpp"
#include "DTK_LibmeshEntityImpl.hpp"

#include <memory>

namespace LibmeshAdapter {
//---------------------------------------------------------------------------//
/*!
 \class LibmeshEntity
 \brief Libmesh entity interface definition.
 */
//---------------------------------------------------------------------------//
template<class LibmeshGeom>
class LibmeshEntity {
public:

	/*!
	 * \brief Constructor.
	 * \param libmesh_element A pointer to the element to wrap this interface
	 * around.
	 */
	LibmeshEntity(const Teuchos::Ptr<LibmeshGeom> &libmesh_object,
			const Teuchos::Ptr<libMesh::MeshBase> &libmesh_mesh,
			const Teuchos::Ptr<LibmeshAdjacencies> &adjacencies) {
		this->b_entity_impl = Teuchos::rcp(
				new LibmeshEntityImpl<LibmeshGeom>(libmesh_object, libmesh_mesh,
						adjacencies));
	}

	LibmeshEntity() { /* ... */
	}

	//---------------------------------------------------------------------------//
	// Copy constructor.
	LibmeshEntity(const LibmeshEntity &rhs) {
		b_entity_impl = rhs.b_entity_impl;
	}

	//---------------------------------------------------------------------------//
	// Copy assignment operator.
	LibmeshEntity &operator=(const LibmeshEntity &rhs) {
		b_entity_impl = rhs.b_entity_impl;
		return *this;
	}

	//---------------------------------------------------------------------------//
	// Move constructor.
	LibmeshEntity(LibmeshEntity &&rhs) {
		b_entity_impl = rhs.b_entity_impl;
	}

	//---------------------------------------------------------------------------//
	// Move assignment operator.
	LibmeshEntity &operator=(LibmeshEntity &&rhs) {
		b_entity_impl = rhs.b_entity_impl;
		return *this;
	}

	//---------------------------------------------------------------------------//
	// brief Destructor.
	~LibmeshEntity() { /* ... */
	}

	//---------------------------------------------------------------------------//
	// Get the unique global identifier for the entity.
	unsigned long int id() const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		return b_entity_impl->id();
	}

	//---------------------------------------------------------------------------//
	// Get the parallel rank that owns the entity.
	int ownerRank() const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		return b_entity_impl->ownerRank();
	}

	//---------------------------------------------------------------------------//
	// Return the topological dimension of the entity.
	int topologicalDimension() const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		return b_entity_impl->topologicalDimension();
	}

	//---------------------------------------------------------------------------//
	// Return the physical dimension of the entity.
	int physicalDimension() const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		return b_entity_impl->physicalDimension();
	}

	//---------------------------------------------------------------------------//
	// Return the Cartesian bounding box around an entity.
	void boundingBox(Teuchos::Tuple<double, 6> &bounds) const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		b_entity_impl->boundingBox(bounds);
	}

	//---------------------------------------------------------------------------//
	// Determine if an entity is in the block with the given id.
	bool inBlock(const int block_id) const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		return b_entity_impl->inBlock(block_id);
	}

	//---------------------------------------------------------------------------//
	// Determine if an entity is on the boundary with the given id.
	bool onBoundary(const int boundary_id) const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		return b_entity_impl->onBoundary(boundary_id);
	}

	//---------------------------------------------------------------------------//
	// Get the extra data on the entity.
	Teuchos::RCP<LibmeshEntityExtraData<LibmeshGeom>> extraData() const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		return b_entity_impl->extraData();
	}

	//---------------------------------------------------------------------------//
	// Provide a one line description of the object.
	std::string description() const {
		DTK_REQUIRE(Teuchos::nonnull(b_entity_impl));
		std::stringstream d;
		d << " Id = " << id() << ", OwnerRank = " << ownerRank()
				<< ", TopologicalDimension = " << topologicalDimension()
				<< ", PhysicalDimension = " << physicalDimension();
		return b_entity_impl->description() + d.str();
	}

	//---------------------------------------------------------------------------//
	// Provide a verbose description of the object.
//    void describe( Teuchos::FancyOStream &out,
//                           const Teuchos::EVerbosityLevel verb_level ) const
//    {
//        DTK_REQUIRE( Teuchos::nonnull( b_entity_impl ) );
//        b_entity_impl->describe( out, verb_level );
//    }

protected:
	// Entity implementation.
	Teuchos::RCP<LibmeshEntityImpl<LibmeshGeom>> b_entity_impl;
};

//---------------------------------------------------------------------------//

}// end namespace LibmeshAdapter

//---------------------------------------------------------------------------//

#endif // end LIBMESHDTKADAPTERS_LIBMESHENTITY_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshEntity.hpp
//---------------------------------------------------------------------------//
