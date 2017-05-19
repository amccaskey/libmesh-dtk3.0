#ifndef LIBMESHUSERAPP_LIBMESHUSERAPPLICATION_HPP
#define LIBMESHUSERAPP_LIBMESHUSERAPPLICATION_HPP

#include <DTK_UserApplication.hpp>
#include <DTK_UserDataInterface.hpp>
#include <DTK_UserFunctionRegistry.hpp>
#include <DTK_View.hpp>

#include <LibmeshManager.hpp>
#include <LibmeshEntity.hpp>
#include <LibmeshEntitySet.hpp>
#include <LibmeshEntityExtraData.hpp>
#include <LibmeshEntityIterator.hpp>

#include <libmesh/dof_map.h>
#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/explicit_system.h>
#include <libmesh/fe.h>
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/node.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/parallel.h>
#include <libmesh/quadrature_gauss.h>
#include <libmesh/system.h>
#include <libmesh/mesh_generation.h>

#include <Kokkos_Core.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <memory>

namespace LibmeshApp {
void nodeListSize(std::shared_ptr<void> user_data, unsigned &space_dim,
		size_t &local_num_nodes, bool &has_ghosts) {

	// Get the LibmeshManager instance
	auto u = std::static_pointer_cast<LibmeshAdapter::LibmeshManager>(
			user_data);

	// Create Predicates to pick out all nodes and local nodes
	auto thisRank = u->entitySet()->communicator()->getRank();
	LibmeshAdapter::NodePredicateFunction localPredicate =
			[=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> e) {return e.ownerRank() == thisRank;};
	LibmeshAdapter::NodePredicateFunction totalPredicate =
			[=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> e) {return true;};

	// Get the Entity Set
	auto entitySet = u->entitySet();

	// Create Iterators over all local nodes and all nodes
	auto localNodeIter = entitySet->entityIterator(localPredicate);
	auto totalNodeIter = entitySet->entityIterator(totalPredicate);

	// Set the total number of local nodes
	local_num_nodes = localNodeIter.size();

	// Set the spacial dimension
	space_dim = entitySet->physicalDimension();

	// Indicate if we have ghosted nodes
	has_ghosts = totalNodeIter.size() != local_num_nodes;
}

void nodeListData(std::shared_ptr<void> user_data,
		DataTransferKit::View<DataTransferKit::Coordinate> coordinates,
		DataTransferKit::View<bool> is_ghost_node) {

	// Get the LibmeshManager instance
	auto u = std::static_pointer_cast<LibmeshAdapter::LibmeshManager>(user_data);

	// Get the EntitySet
	auto entitySet = u->entitySet();

	// Get the spacial dimenstion
	auto dim = entitySet->physicalDimension();

	// Create a predicate that picks out only local nodes
	auto thisRank = u->entitySet()->communicator()->getRank();
	LibmeshAdapter::NodePredicateFunction localPredicate =
			[=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> e) {return e.ownerRank() == thisRank;};

	// Create the entity iterator over those local nodes
	auto localNodeIter = entitySet->entityIterator(localPredicate);

	// Loop over all nodes and set their spatial coordinates
	unsigned num_nodes = localNodeIter.size();
	unsigned counter = 0;
	auto startNode = localNodeIter.begin();
	auto endNode = localNodeIter.end();
	for (auto node = startNode; node != endNode; ++node) {
		auto libmeshNode = Teuchos::rcp_dynamic_cast<
				LibmeshAdapter::LibmeshEntityExtraData<libMesh::Node>>(
				node->extraData());
		for (unsigned d = 0; d < dim; ++d) {
			coordinates[num_nodes * d + counter] =
					libmeshNode->d_libmesh_geom->operator()(d);
		}

		counter++;
	}
}

void boundingVolumeListSize( std::shared_ptr<void> user_data, unsigned &space_dim,
                        size_t &local_num_volumes, bool &has_ghosts ) {

}

//---------------------------------------------------------------------------//
/*
 * \brief Get the data for a bounding volume list.
 */
void boundingVolumeData(
    std::shared_ptr<void> user_data, DataTransferKit::View<DataTransferKit::Coordinate> bounding_volumes,
	DataTransferKit::View<bool> is_ghost_volume ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size parameters for building a polyhedron list.
 */
void polyhedronListSizeFunction( std::shared_ptr<void> user_data, unsigned &space_dim,
                        size_t &local_num_nodes, size_t &local_num_faces,
                        size_t &total_nodes_per_face, size_t &local_num_cells,
                        size_t &total_faces_per_cell, bool &has_ghosts ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the data for a polyhedron list.
 */
void polyhedronListDataFunction(
    std::shared_ptr<void> user_data, DataTransferKit::View<DataTransferKit::Coordinate> coordinates,
	DataTransferKit::View<DataTransferKit::LocalOrdinal> faces, DataTransferKit::View<unsigned> nodes_per_face,
	DataTransferKit::View<DataTransferKit::LocalOrdinal> cells, DataTransferKit::View<unsigned> faces_per_cell,
	DataTransferKit::View<int> face_orientation, DataTransferKit::View<bool> is_ghost_cell ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size parameters for building a cell list with a single
 * topology.
 */
void cellListSizeFunction( std::shared_ptr<void> user_data, unsigned &space_dim,
                        size_t &local_num_nodes, size_t &local_num_cells,
                        unsigned &nodes_per_cell, bool &has_ghosts ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the data for a single topology cell list.
 */
void cellListDataFunction( std::shared_ptr<void> user_data,
		DataTransferKit::View<DataTransferKit::Coordinate> coordinates, DataTransferKit::View<DataTransferKit::LocalOrdinal> cells,
		DataTransferKit::View<bool> is_ghost_cell, std::string &cell_topology ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size parameters for building a cell list with mixed
 * topologies.
 */
void mixedTopologyCellListSizeFunction( std::shared_ptr<void> user_data, unsigned &space_dim,
                        size_t &local_num_nodes, size_t &local_num_cells,
                        size_t &total_nodes_per_cell, bool &has_ghosts ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the data for a mixed topology cell list.
 */
void mixedTopologyCellListDataFunction(
    std::shared_ptr<void> user_data, DataTransferKit::View<DataTransferKit::Coordinate> coordinates,
	DataTransferKit::View<DataTransferKit::LocalOrdinal> cells, DataTransferKit::View<unsigned> cell_topology_ids,
	DataTransferKit::View<bool> is_ghost_cell, std::vector<std::string> &cell_topologies ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size parameters for a boundary.
 */
void boundarySizeFunctio(
    std::shared_ptr<void> user_data, size_t &local_num_faces ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the data for a boundary.
 */
void boundaryDataFunction(
    std::shared_ptr<void> user_data, DataTransferKit::View<DataTransferKit::LocalOrdinal> boundary_cells,
	DataTransferKit::View<unsigned> cell_faces_on_boundary ) {

}

//---------------------------------------------------------------------------//
// Degree-of-freedom interface.
//---------------------------------------------------------------------------//
/*!
 * \brief Get the size parameters for a degree-of-freedom id map with a single
 * number of dofs per object.
 */
void dofMapSizeFunction( std::shared_ptr<void> user_data, size_t &local_num_dofs,
                        size_t &local_num_objects, unsigned &dofs_per_object ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the data for a degree-of-freedom id map with a single number of
 * dofs per object.
 */
void dofMapDataFunction(
    std::shared_ptr<void> user_data, DataTransferKit::View<DataTransferKit::GlobalOrdinal> global_dof_ids,
	DataTransferKit::View<DataTransferKit::LocalOrdinal> object_dof_ids, std::string &discretization_type ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size parameters for a degree-of-freedom id map with each
 * object having a potentially different number of dofs (e.g. mixed topology
 * cell lists or polyhedron lists).
 */
void mixedTopologyDOFMapSizeFunctio(
    std::shared_ptr<void> user_data, size_t &local_num_dofs,
    size_t &local_num_objects, size_t &total_dofs_per_object ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the data for a multiple object degree-of-freedom id map
 * (e.g. mixed topology cell lists or polyhedron lists).
 */
void mixedTopologyDOFMapDataFunction(
    std::shared_ptr<void> user_data, DataTransferKit::View<DataTransferKit::GlobalOrdinal> global_dof_ids,
	DataTransferKit::View<DataTransferKit::LocalOrdinal> object_dof_ids, DataTransferKit::View<unsigned> dofs_per_object,
    std::string &discretization_type ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size parameters for a field. Field must be of size
 * local_num_dofs in the associated dof_id_map.
 */
template <class Scalar>
void fieldSizeFunction( std::shared_ptr<void> user_data,
                        unsigned &field_dimension, size_t &local_num_dofs ) {

}

//---------------------------------------------------------------------------//
/*!
 * \brief Pull data from application into a field.
 */
template <class Scalar>
void pullFieldDataFunction(
    std::shared_ptr<void> user_data, DataTransferKit::View<Scalar> field_dofs ) {

}

//---------------------------------------------------------------------------//
/*
 * \brief Push data from a field into the application.
 */
template <class Scalar>
void pushFieldDataFunction(
    std::shared_ptr<void> user_data, const DataTransferKit::View<Scalar> field_dofs ) {

}

//---------------------------------------------------------------------------//
/*
 * \brief Evaluate a field at a given set of points in a given set of objects.
 */
template <class Scalar>
void evaluateFieldFunction(
    std::shared_ptr<void> user_data, const DataTransferKit::View<DataTransferKit::Coordinate> evaluation_points,
    const DataTransferKit::View<DataTransferKit::LocalOrdinal> object_ids, DataTransferKit::View<Scalar> values ) {

}


}

#endif
