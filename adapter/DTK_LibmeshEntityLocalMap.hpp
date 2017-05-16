//---------------------------------------------------------------------------//
/*
 Copyright (c) 2012, Stuart R. Slattery
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

 *: Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 *: Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.

 *: Neither the name of the University of Wisconsin - Madison nor the
 names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
//---------------------------------------------------------------------------//
/*!
 * \brief DTK_LibmeshEntityLocalMap.hpp
 * \author Stuart R. Slattery
 * \brief Forward and reverse local mappings for entities.
 */
//---------------------------------------------------------------------------//
#ifndef LIBMESHDTKADAPTERS_LIBMESHENTITYLOCALMAP_HPP
#define LIBMESHDTKADAPTERS_LIBMESHENTITYLOCALMAP_HPP

#include "DTK_LibmeshEntityExtraData.hpp"

#include "DTK_EntityLocalMap.hpp"
#include "DTK_DBC.hpp"

#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <libmesh/mesh_base.h>
#include <libmesh/system.h>

namespace LibmeshAdapter {
//---------------------------------------------------------------------------//
/*!
 \class LibmeshEntityLocalMap
 \brief Libmesh mesh forward and reverse local map implementation.
 */
//---------------------------------------------------------------------------//
class LibmeshEntityLocalMap {
public:
	/*!
	 * \brief Constructor.
	 */
	LibmeshEntityLocalMap(const Teuchos::RCP<libMesh::MeshBase> &libmesh_mesh,
			const Teuchos::RCP<libMesh::System> &libmesh_system) :
			d_libmesh_mesh(libmesh_mesh), d_libmesh_system(libmesh_system), d_newton_tol(
					1.0e-9), d_inclusion_tol(1.0e-6) {
	}

	/*
	 * \brief Set parameters for mapping.
	 * \param parameters Parameters for mapping.
	 */
	void setParameters(const Teuchos::ParameterList &parameters) {
		if (parameters.isParameter("Point Inclusion Tolerance")) {
			d_inclusion_tol = parameters.get<double>(
					"Point Inclusion Tolerance");
		}
		if (parameters.isParameter("Newton Tolerance")) {
			d_newton_tol = parameters.get<double>("Newton Tolerance");
		}
	}

	/*!
	 * \brief Return the entity measure with respect to the parameteric
	 * dimension (volume for a 3D entity, area for 2D, and length for 1D).
	 * \param entity Compute the measure for this entity.
	 * \return The measure of the entity.
	 */
	template<typename LibmeshEntityT>
	double measure(const LibmeshEntityT &entity) const {
		if (0 == entity.topologicalDimension()) {
			return 0.0;
		}

		return LibmeshHelpers::extractGeom<libMesh::Elem>(entity)->volume();
	}

	/*!
	 * \brief Return the centroid of the entity.
	 * \param centroid A view of the centroid coordinates. This view will
	 * be allocated. Assign a view of your centroid to this view.
	 */
	template<typename LibmeshEntityT>
	void centroid(const LibmeshEntityT &entity,
			const Teuchos::ArrayView<double> &centroid) const {
		libMesh::Point point;

		if (0 == entity.topologicalDimension()) {
			Teuchos::Ptr<libMesh::Point> node = LibmeshHelpers::extractGeom<
					libMesh::Node>(entity);
			point = *node;
		} else {
			point =
					LibmeshHelpers::extractGeom<libMesh::Elem>(entity)->centroid();
		}

		int space_dim = entity.physicalDimension();
		for (int d = 0; d < space_dim; ++d) {
			centroid[d] = point(d);
		}

	}

	/*!
	 * \brief (Safeguard the reverse map) Perform a safeguard check for
	 * mapping a point to the reference space of an entity using the given
	 * tolerance.
	 * \param entity Perfrom the mapping for this entity.
	 * \param parameters Parameters to be used for the safeguard check.
	 * \param physical_point A view into an array of size physicalDimension()
	 * containing the coordinates of the point to map.
	 * \return Return true if it is safe to map to the reference frame.
	 */
	template<typename LibmeshEntityT>
	bool isSafeToMapToReferenceFrame(const LibmeshEntityT &entity,
			const Teuchos::ArrayView<const double> &physical_point) const {
		int space_dim = entity.physicalDimension();
		int param_dim = 0;

		if (0 != entity.topologicalDimension()) {
			param_dim =
					LibmeshHelpers::extractGeom<libMesh::Elem>(entity)->dim();
		} else {
			return false;
		}

		if (space_dim == param_dim) {

			// Get the bounding box of the entity.
			Teuchos::Tuple<double, 6> entity_box;
			entity.boundingBox(entity_box);

			// Check if the point is in the bounding box of the entity.
			double tolerance = 1.0e-6;
			int space_dim = entity.physicalDimension();
			bool in_x = true;
			if (space_dim > 0) {
				double x_tol = (entity_box[3] - entity_box[0]) * tolerance;
				in_x = ((physical_point[0] >= (entity_box[0] - x_tol))
						&& (physical_point[0] <= (entity_box[3] + x_tol)));
			}
			bool in_y = true;
			if (space_dim > 1) {
				double y_tol = (entity_box[4] - entity_box[1]) * tolerance;
				in_y = ((physical_point[1] >= (entity_box[1] - y_tol))
						&& (physical_point[1] <= (entity_box[4] + y_tol)));
			}
			bool in_z = true;
			if (space_dim > 2) {
				double z_tol = (entity_box[5] - entity_box[2]) * tolerance;
				in_z = ((physical_point[2] >= (entity_box[2] - z_tol))
						&& (physical_point[2] <= (entity_box[5] + z_tol)));
			}
			bool x = (in_x && in_y && in_z);

			// See if we are in the Cartesian bounding box.
			if (x) {
				// If we are in the Cartesian bounding box see if we are 'close'
				// to the element according to libMesh.
				int space_dim = entity.physicalDimension();
				libMesh::Point lm_point;
				for (int d = 0; d < space_dim; ++d) {
					lm_point(d) = physical_point[d];
				}
				return LibmeshHelpers::extractGeom<libMesh::Elem>(entity)->close_to_point(
						lm_point, d_inclusion_tol);
			} else {
				return false;
			}
		} else {
			// We currently do not natively support checks for mapping to
			// surfaces.
			bool not_implemented = true;
			assert(!not_implemented);
		}
		return false;
	}

	/*!
	 * \brief (Reverse Map) Map a point to the reference space of an
	 * entity. Return the parameterized point.
	 * \param entity Perfrom the mapping for this entity.
	 * \param parameters Parameters to be used for the mapping procedure.
	 * \param physical_point A view into an array of size physicalDimension()
	 * containing the coordinates of the point to map.
	 * \param reference_point A view into an array of size physicalDimension()
	 * to write the reference coordinates of the mapped point.
	 * \return Return true if the map to reference frame succeeded.
	 */
	template<typename LibmeshEntityT>
	bool mapToReferenceFrame(const LibmeshEntityT &entity,
			const Teuchos::ArrayView<const double> &physical_point,
			const Teuchos::ArrayView<double> &reference_point) const {
		int space_dim = entity.physicalDimension();
		libMesh::Point lm_point;
		for (int d = 0; d < space_dim; ++d) {
			lm_point(d) = physical_point[d];
		}

		libMesh::Point lm_reference_point = libMesh::FEInterface::inverse_map(
				space_dim, d_libmesh_system->variable_type(0),
				LibmeshHelpers::extractGeom<libMesh::Elem>(entity).getRawPtr(),
				lm_point, d_newton_tol);

		for (int d = 0; d < space_dim; ++d) {
			reference_point[d] = lm_reference_point(d);
		}

		return true;
	}

	/*!
	 * \brief Determine if a reference point is in the parameterized space of
	 * an entity.
	 * \param entity Perfrom the mapping for this entity.
	 * \param parameters Parameters to be used for the point inclusion check.
	 * \param reference_point A view into an array of size physicalDimension()
	 * containing the reference coordinates of the mapped point.
	 * \return True if the point is in the reference space, false if not.
	 */
	template<typename LibmeshEntityT>
	bool checkPointInclusion(const LibmeshEntityT &entity,
			const Teuchos::ArrayView<const double> &reference_point) const {
		int space_dim = entity.physicalDimension();
		libMesh::Point lm_reference_point;
		for (int d = 0; d < space_dim; ++d) {
			lm_reference_point(d) = reference_point[d];
		}

		return libMesh::FEInterface::on_reference_element(lm_reference_point,
				LibmeshHelpers::extractGeom<libMesh::Elem>(entity)->type(),
				d_inclusion_tol);
	}

	/*!
	 * \brief (Forward Map) Map a reference point to the physical space of an
	 * entity.
	 * \param entity Perfrom the mapping for this entity.
	 * \param reference_point A view into an array of size physicalDimension()
	 * containing the reference coordinates of the mapped point.
	 * \param physical_point A view into an array of size physicalDimension()
	 * to write the coordinates of physical point.
	 */
	template<typename LibmeshEntityT>
	void mapToPhysicalFrame(const LibmeshEntityT &entity,
			const Teuchos::ArrayView<const double> &reference_point,
			const Teuchos::ArrayView<double> &physical_point) const {
		int space_dim = entity.physicalDimension();
		libMesh::Point lm_reference_point;
		for (int d = 0; d < space_dim; ++d) {
			lm_reference_point(d) = reference_point[d];
		}

		libMesh::Point lm_point = libMesh::FEInterface::map(space_dim,
				d_libmesh_system->variable_type(0),
				LibmeshHelpers::extractGeom<libMesh::Elem>(entity).getRawPtr(),
				lm_reference_point);

		for (int d = 0; d < space_dim; ++d) {
			physical_point[d] = lm_point(d);
		}
	}

	/*!
	 * \brief Compute the normal on a face (3D) or edge (2D) at a given
	 * reference point. A default implementation is provided using a finite
	 * difference scheme.
	 * \param entity Compute the normal for this entity.
	 * \param parent_entity The adjacent parent entity used to determine which
	 * direction is outward. The parent entity should be of a higher
	 * topological dimension than the entity and be adjacent to the entity.
	 * \param reference_point Compute the normal at this reference point.
	 * \param normal A view into an array of size physicalDimension() to write
	 * the normal.
	 */
	template<typename LibmeshEntityT, typename LibmeshEntityS>
	void normalAtReferencePoint(const LibmeshEntityT &entity,
			const LibmeshEntityS &parent_entity,
			const Teuchos::ArrayView<const double> &reference_point,
			const Teuchos::ArrayView<double> &normal) const {
		// Determine the reference dimension.
		int physical_dim = entity.physicalDimension();
		int ref_dim = physical_dim - 1;

		// Create a perturbation.
		double perturbation = std::sqrt(std::numeric_limits<double>::epsilon());

		// 3D/face case.
		if (2 == ref_dim) {
			DTK_CHECK(3 == reference_point.size());
			DTK_CHECK(3 == normal.size());

			// Create extra points.
			Teuchos::Array<double> ref_p1(reference_point);
			Teuchos::Array<double> ref_p2(reference_point);

			// Apply a perturbation to the extra points.
			double p1_sign = 1.0;
			ref_p1[0] += perturbation;
			if (!this->checkPointInclusion(entity, ref_p1())) {
				ref_p1[0] -= 2 * perturbation;
				p1_sign = -1.0;
			}
			double p2_sign = 1.0;
			ref_p2[1] += perturbation;
			if (!this->checkPointInclusion(entity, ref_p2())) {
				ref_p2[1] -= 2 * perturbation;
				p2_sign = -1.0;
			}

			// Map the perturbed points to the physical frame.
			Teuchos::Array<double> p0(physical_dim);
			this->mapToPhysicalFrame(entity, reference_point(), p0());
			Teuchos::Array<double> p1(physical_dim);
			this->mapToPhysicalFrame(entity, ref_p1(), p1());
			Teuchos::Array<double> p2(physical_dim);
			this->mapToPhysicalFrame(entity, ref_p2(), p2());

			// Compute the cross product of the tangents produced by the
			// perturbation.
			Teuchos::Array<double> tan1(physical_dim);
			Teuchos::Array<double> tan2(physical_dim);
			for (int d = 0; d < physical_dim; ++d) {
				tan1[d] = p1_sign * (p1[d] - p0[d]);
				tan2[d] = p2_sign * (p2[d] - p0[d]);
			}
			normal[0] = tan1[1] * tan2[2] - tan1[2] * tan2[1];
			normal[1] = tan1[2] * tan2[0] - tan1[0] * tan2[2];
			normal[2] = tan1[0] * tan2[1] - tan1[1] * tan2[0];
		}

		// 2D/edge case.
		else if (1 == ref_dim) {
			DTK_CHECK(2 == reference_point.size());
			DTK_CHECK(2 == normal.size());

			// Create extra points.
			Teuchos::Array<double> ref_p1(reference_point);

			// Apply a perturbation to the extra points.
			double p1_sign = 1.0;
			ref_p1[0] += perturbation;
			if (!this->checkPointInclusion(entity, ref_p1())) {
				ref_p1[0] -= 2 * perturbation;
				p1_sign = -1.0;
			}

			// Map the perturbed points to the physical frame.
			Teuchos::Array<double> p0(physical_dim);
			this->mapToPhysicalFrame(entity, reference_point(), p0());
			Teuchos::Array<double> p1(physical_dim);
			this->mapToPhysicalFrame(entity, ref_p1(), p1());

			// Compute the cross product of the tangents produced by the
			// perturbation.
			Teuchos::Array<double> tan(physical_dim);
			for (int d = 0; d < physical_dim; ++d) {
				tan[d] = p1_sign * (p1[d] - p0[d]);
			}
			normal[0] = -tan[0];
			normal[1] = tan[1];
		}

		// Normalize the normal vector.
		double norm = 0.0;
		for (int d = 0; d < physical_dim; ++d) {
			norm += normal[d] * normal[d];
		}
		norm = std::sqrt(norm);
		for (int d = 0; d < physical_dim; ++d) {
			normal[d] /= norm;
		}
	}

private:
	// Libmesh mesh.
	Teuchos::RCP<libMesh::MeshBase> d_libmesh_mesh;

	// Libmesh system.
	Teuchos::RCP<libMesh::System> d_libmesh_system;

	// Newton tolerance.
	double d_newton_tol;

	// Point inclusion tolerance.
	double d_inclusion_tol;
};

//---------------------------------------------------------------------------//

}// end namespace LibmeshAdapter

#endif // end LIBMESHDTKADAPTERS_LIBMESHENTITYLOCALMAP_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshEntityLocalMap.hpp
//---------------------------------------------------------------------------//
