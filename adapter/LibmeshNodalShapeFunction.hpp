#ifndef LIBMESHDTKADAPTERS_LIBMESHNODALSHAPEFUNCTION
#define LIBMESHDTKADAPTERS_LIBMESHNODALSHAPEFUNCTION

#include "LibmeshEntityExtraData.hpp"

#include <Types.hpp>
#include "DTK_DBC.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include <libmesh/mesh_base.h>
#include <libmesh/system.h>
#include <libmesh/fe_interface.h>
#include <libmesh/fe_compute_data.h>
namespace LibmeshAdapter
{
typedef unsigned long int SupportId;

//---------------------------------------------------------------------------//
/*!
  \class LibmeshNodalShapeFunction
  \brief Nodal shape function implementation for Libmesh mesh.

  LibmeshNodalShapeFunction provides a shape function for node-centered
  quantities with shape functions evaluated in an element supported by
  nodes. The node ids serve as the dof ids for these shape functions. A
  corresponding DOF vector indexed via node ids should be produced to match
  this shape function. LibmeshDOFVector provides services to construct these
  vectors.
*/
//---------------------------------------------------------------------------//
class LibmeshNodalShapeFunction
{
  public:
    /*!
     * \brief Constructor.
     */
	LibmeshNodalShapeFunction(
			const Teuchos::RCP<libMesh::MeshBase> &libmesh_mesh,
			const Teuchos::RCP<libMesh::System> &libmesh_system) :
			d_libmesh_mesh(libmesh_mesh), d_libmesh_system(libmesh_system) {
	}

    /*!
     * \brief Given an entity, get the ids of its support locations.
     * \param entity Get the support locations for this entity.
     * \param support_ids Return the ids of the degrees of freedom in the
     * parallel
     * vector space supporting the entities.
     */
	template<typename LibmeshEntityT>
    void entitySupportIds( const LibmeshEntityT &entity,
                           Teuchos::Array<SupportId> &support_ids ) const;

    /*!
     * \brief Given an entity and a reference point, evaluate the shape
     * function of the entity at that point.
     * \param entity Evaluate the shape function of this entity.
     * \param reference_point Evaluate the shape function at this point
     * given in reference coordinates.
     * \param values Entity shape function evaluated at the reference
     * point.
     */
	template<typename LibmeshEntityT>
    void evaluateValue( const LibmeshEntityT &entity,
                        const Teuchos::ArrayView<const double> &reference_point,
                        Teuchos::Array<double> &values ) const {
	    int space_dim = entity.physicalDimension();
	    libMesh::Point lm_reference_point;
	    for ( int d = 0; d < space_dim; ++d )
	    {
	        lm_reference_point( d ) = reference_point[d];
	    }

	    libMesh::FEComputeData fe_compute_data(
	        d_libmesh_system->get_equation_systems(), lm_reference_point );

	    libMesh::FEInterface::compute_data(
	        space_dim, d_libmesh_system->variable_type( 0 ),
	        extractGeom<libMesh::Elem>( entity ).getRawPtr(), fe_compute_data );

	    values = fe_compute_data.shape;
	}

    /*!
     * \brief Given an entity and a reference point, evaluate the gradient of
     * the shape function of the entity at that point.
     * \param entity Evaluate the shape function of this entity.
     * \param reference_point Evaluate the shape function at this point
     * given in reference coordinates.
     * \param gradients Entity shape function gradients evaluated at the
     * reference point. Return these ordered with respect to those return by
     * getDOFIds() such that gradients[N][D] gives the gradient value of the
     * Nth DOF in the Dth spatial dimension.
     */
	template<typename LibmeshEntityT>
    void
    evaluateGradient( const LibmeshEntityT &entity,
                      const Teuchos::ArrayView<const double> &reference_point,
                      Teuchos::Array<Teuchos::Array<double>> &gradients ) const {
		// FIXME NOT SURE WHY THIS IS EMPTY
	}

  private:
    // Extract the libmesh geom object.
    template <class LibmeshGeom>
    Teuchos::Ptr<LibmeshGeom> extractGeom( const LibmeshEntity<LibmeshGeom> &entity ) const;

  private:
    // Libmesh mesh.
    Teuchos::RCP<libMesh::MeshBase> d_libmesh_mesh;

    // Libmesh system.
    Teuchos::RCP<libMesh::System> d_libmesh_system;
};

//---------------------------------------------------------------------------//
// Template functions.
//---------------------------------------------------------------------------//
// Extract the libmesh geom object.
template<class LibmeshGeom>
Teuchos::Ptr<LibmeshGeom> LibmeshNodalShapeFunction::extractGeom(
		const LibmeshEntity<LibmeshGeom> &entity) const {
	return Teuchos::rcp_dynamic_cast<LibmeshEntityExtraData<LibmeshGeom>>(
			entity.extraData())->d_libmesh_geom;
}

//---------------------------------------------------------------------------//

template<>
void LibmeshNodalShapeFunction::entitySupportIds(
		const LibmeshEntity<libMesh::Node> &entity,
		Teuchos::Array<SupportId> &support_ids) const {
	// Node case.
	DTK_CHECK(extractGeom<libMesh::Node>(entity)->valid_id());
	support_ids.assign(1, extractGeom<libMesh::Node>(entity)->id());
}

template<>
void LibmeshNodalShapeFunction::entitySupportIds(
		const LibmeshEntity<libMesh::Elem> &entity,
		Teuchos::Array<SupportId> &support_ids) const {
	Teuchos::Ptr<libMesh::Elem> elem = extractGeom<libMesh::Elem>(entity);
	int num_nodes = elem->n_nodes();
	support_ids.resize(num_nodes);
	for (int n = 0; n < num_nodes; ++n) {
		DTK_CHECK(elem->get_node(n)->valid_id());
		support_ids[n] = elem->get_node(n)->id();
	}
}

} // end namespace LibmeshAdapter

//---------------------------------------------------------------------------//

#endif // end LIBMESHDTKADPATERS_LIBMESHNODALSHAPEFUNCTION

//---------------------------------------------------------------------------//
// end DTK_LibmeshNodalShapeFunction.hpp
//---------------------------------------------------------------------------//
