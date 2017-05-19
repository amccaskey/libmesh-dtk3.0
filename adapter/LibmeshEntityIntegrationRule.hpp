#ifndef DTK_LIBMESHENTITYINTEGRATIONRULE_HPP
#define DTK_LIBMESHENTITYINTEGRATIONRULE_HPP

#include <map>

#include <Teuchos_Array.hpp>
#include "LibmeshHelpers.hpp"

#include <libmesh/quadrature.h>

namespace LibmeshAdapter
{
//---------------------------------------------------------------------------//
/*!
  \class LibmeshEntityIntegrationRule
  \brief integration rule interface.

  LibmeshEntityIntegrationRule provides numerical quadrature for entities.
*/
//---------------------------------------------------------------------------//
class LibmeshEntityIntegrationRule
{
  public:
    /*
     * \brief Constructor.
     */
    LibmeshEntityIntegrationRule(
        const libMesh::QuadratureType quadrature_type = libMesh::QGAUSS ) : d_quad_type( quadrature_type ) {

    }

    /*!
     * \brief Given an entity and an integration order, get its integration
     * rule.
     *
     * \param entity Get the integration rule for this entity.
     *
     * \param order Get an integration rule of this order.
     *
     * \param reference_points Return the integration points in the reference
     * frame of the entity in this array. If there are N integration points of
     * topological dimension D then this array is of size
     * reference_points[N][D].
     *
     * \param weights Return the weights of the integration points in this
     * array. If there are N integration points this array is of size
     * weights[N].
     */
    template<typename LibmeshEntityT>
    void getIntegrationRule(
        const LibmeshEntityT &entity, const int order,
			Teuchos::Array<Teuchos::Array<double>> &reference_points,
			Teuchos::Array<double> &weights) const {
		// Create a libmesh quadrature rule. Use Gauss quadrature as the default.
		libMesh::UniquePtr<libMesh::QBase> libmesh_quadrature =
				libMesh::QBase::build(d_quad_type,
						entity.topologicalDimension(),
						static_cast<libMesh::Order>(order));

		// Initialize the quadrature rule for the entity.
		Teuchos::Ptr<libMesh::Elem> libmesh_elem = LibmeshHelpers::extractGeom<
				libMesh::Elem>(entity);
		libmesh_quadrature->init(libmesh_elem->type());

		// Extract the data for the quadrature rule.
		int num_points = libmesh_quadrature->n_points();
		int quad_dim = libmesh_quadrature->get_dim();
		reference_points.resize(num_points);
		weights.resize(num_points);
		for (int p = 0; p < num_points; ++p) {
			weights[p] = libmesh_quadrature->w(p);
			libMesh::Point qp = libmesh_quadrature->qp(p);
			reference_points[p].resize(quad_dim);
			for (int d = 0; d < quad_dim; ++d) {
				reference_points[p][d] = qp(d);
			}
		}
    }

  private:
    // libMesh quadrature type.
    libMesh::QuadratureType d_quad_type;
};

//---------------------------------------------------------------------------//

} // end namespace LibmeshAdapter

//---------------------------------------------------------------------------//

#endif // end DTK_LIBMESHENTITYINTEGRATIONRULE_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshEntityIntegrationRule.hpp
//---------------------------------------------------------------------------//
