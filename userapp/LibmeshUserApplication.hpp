#ifndef LIBMESHUSERAPP_LIBMESHUSERAPPLICATION_HPP
#define LIBMESHUSERAPP_LIBMESHUSERAPPLICATION_HPP

#include <DTK_ConfigDefs.hpp>
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

// nodes, elems, and fields give us 3 things - point cloud transfer, finite element interpolation

namespace LibmeshApp {

class LibmeshUserApplication {

protected:

	int thisRank = 0;

	std::shared_ptr<LibmeshAdapter::LibmeshManager> manager;

	Teuchos::RCP<LibmeshAdapter::LibmeshEntitySet> entitySet;

public:

	LibmeshUserApplication(std::shared_ptr<LibmeshAdapter::LibmeshManager> user_data) :
			manager(user_data), entitySet(user_data->entitySet()), thisRank(user_data->entitySet()->communicator()->getRank()) {

		entitySet = manager->entitySet();

		thisRank = entitySet->communicator()->getRank();

	}

	void nodeListSize(std::shared_ptr<void> user_data, unsigned &space_dim,
			size_t &local_num_nodes, bool &has_ghosts) {

		// Create Predicates to pick out all nodes and local nodes
		LibmeshAdapter::NodePredicateFunction localPredicate =
				[=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> e) {return e.ownerRank() == thisRank;};
		LibmeshAdapter::NodePredicateFunction totalPredicate =
				[=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> e) {return true;};

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

		bool has_ghosts = false;
		unsigned space_dim = 0;
		size_t local_num_nodes = 0;
		nodeListSize(user_data, space_dim, local_num_nodes, has_ghosts);

		// Create a predicate that picks out only local nodes
		auto thisRank = entitySet->communicator()->getRank();
		LibmeshAdapter::NodePredicateFunction localPredicate =
				[=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> e) {return e.ownerRank() == thisRank;};

		// Create the entity iterator over those local nodes
		auto localNodeIter = entitySet->entityIterator(localPredicate);

		// Loop over all nodes and set their spatial coordinates
		unsigned counter = 0;
		auto startNode = localNodeIter.begin();
		auto endNode = localNodeIter.end();
		for (auto node = startNode; node != endNode; ++node) {
			auto libmeshNode = Teuchos::rcp_dynamic_cast<
					LibmeshAdapter::LibmeshEntityExtraData<libMesh::Node>>(
					node->extraData());
			for (unsigned d = 0; d < space_dim; ++d) {
				coordinates[local_num_nodes * d + counter] =
						libmeshNode->d_libmesh_geom->operator()(d);
				if (has_ghosts) {
					is_ghost_node[counter] =
							libmeshNode->d_libmesh_geom->processor_id()
									== thisRank;
				}
			}

			counter++;
		}
	}

	// CELLS ARE Elements. list of cells that all have same topology.
	void cellListSize(std::shared_ptr<void> user_data, unsigned &space_dim,
			size_t &local_num_nodes, size_t &local_num_cells,
			unsigned &nodes_per_cell, bool &has_ghosts) {

		// Create a predicate that picks out only local cells
		LibmeshAdapter::ElemPredicateFunction localPredicate =
				[=]( LibmeshAdapter::LibmeshEntity<libMesh::Elem> e) {return e.ownerRank() == thisRank;};
		LibmeshAdapter::ElemPredicateFunction totalPredicate =
				[=]( LibmeshAdapter::LibmeshEntity<libMesh::Elem> e) {return true;};

		// Create the entity iterator over those local cells
		auto localElemIter = entitySet->entityIterator(localPredicate);
		auto totalElemIter = entitySet->entityIterator(totalPredicate);

		// Get the number of local nodes and space dimension
		bool has_node_ghosts = false;
		nodeListSize(user_data, space_dim, local_num_nodes, has_node_ghosts);

		// Set the local number of cells and if there are ghosts
		local_num_cells = localElemIter.size();
		has_ghosts = totalElemIter.size() != local_num_cells;

		// Get the total number of nodes per cell
		auto startElem = localElemIter.begin();
		auto libmeshElem = Teuchos::rcp_dynamic_cast<
				LibmeshAdapter::LibmeshEntityExtraData<libMesh::Elem>>(
				startElem->extraData());
		nodes_per_cell = libmeshElem->d_libmesh_geom->n_nodes();
	}

	//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the data for a single topology cell list.
	 */
	void cellListData(std::shared_ptr<void> user_data,
			DataTransferKit::View<DataTransferKit::Coordinate> coordinates,
			DataTransferKit::View<DataTransferKit::LocalOrdinal> cells,
			DataTransferKit::View<bool> is_ghost_cell,
			std::string &cell_topology) {

		// coordinates are the coords of the cells' nodes.
		unsigned space_dim = 0, nodes_per_cell = 0;
		size_t local_num_nodes = 0, local_num_cells = 0;
		bool has_ghosts = false;
		cellListSize(user_data, space_dim, local_num_nodes, local_num_cells,
				nodes_per_cell, has_ghosts);

		// Get the node list data
		Kokkos::View<bool *, Kokkos::LayoutLeft, Kokkos::Serial> ghostsArray(
				"is_ghost_node", local_num_nodes);
		DataTransferKit::View<bool> is_ghost_node(ghostsArray);
		nodeListData(user_data, coordinates, is_ghost_node);

		// Create a predicate that picks out only local cells
		LibmeshAdapter::ElemPredicateFunction localPredicate =
				[=]( LibmeshAdapter::LibmeshEntity<libMesh::Elem> e) {return e.ownerRank() == thisRank;};
		// Create the entity iterator over those local cells
		auto localElemIter = entitySet->entityIterator(localPredicate);

		// Loop over all nodes and set their spatial coordinates
		unsigned elemCounter = 0;
		auto startElem = localElemIter.begin();
		auto endElem = localElemIter.end();
		for (auto elem = startElem; elem != endElem; ++elem) {
			auto libmeshElem = Teuchos::rcp_dynamic_cast<
					LibmeshAdapter::LibmeshEntityExtraData<libMesh::Elem>>(
					elem->extraData());

			if (has_ghosts) {
				is_ghost_cell[elemCounter] =
						libmeshElem->d_libmesh_geom->processor_id() == thisRank;
			}

			for (unsigned i = 0; i < nodes_per_cell; ++i) {
//				std::cout << "cells[" << elemCounter << "*8+" << i << "="
//						<< (elemCounter * nodes_per_cell + i) << "] = "
//						<< libmeshElem->d_libmesh_geom->node(i) << "\n";
				cells[elemCounter * nodes_per_cell + i] =
						libmeshElem->d_libmesh_geom->node(i);
			}

			elemCounter++;
		}

	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the size parameters for building a cell list with mixed
	 * topologies.
	 */
	void mixedTopologyCellListSizeFunction(std::shared_ptr<void> user_data,
			unsigned &space_dim, size_t &local_num_nodes,
			size_t &local_num_cells, size_t &total_nodes_per_cell,
			bool &has_ghosts) {

	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the data for a mixed topology cell list.
	 */
	void mixedTopologyCellListDataFunction(std::shared_ptr<void> user_data,
			DataTransferKit::View<DataTransferKit::Coordinate> coordinates,
			DataTransferKit::View<DataTransferKit::LocalOrdinal> cells,
			DataTransferKit::View<unsigned> cell_topology_ids,
			DataTransferKit::View<bool> is_ghost_cell,
			std::vector<std::string> &cell_topologies) {

	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the size parameters for a boundary.
	 */
	void boundarySizeFunction(std::shared_ptr<void> user_data, const std::string& boundary_name,
			size_t &local_num_faces) {

		auto boundaryInfo = manager->getMesh()->get_boundary_info();
		auto boundaryId = boundaryInfo.get_id_by_name(boundary_name);
		local_num_faces = 0;

		// Get all local elements
		LibmeshAdapter::ElemPredicateFunction localPredicate =
						[=]( LibmeshAdapter::LibmeshEntity<libMesh::Elem> e) {return e.ownerRank() == thisRank;};


		// Create the entity iterator over those local cells
		auto localElemIter = entitySet->entityIterator(localPredicate);

		// Loop over all nodes and set their spatial coordinates
		auto startElem = localElemIter.begin();
		auto endElem = localElemIter.end();
		for (auto elem = startElem; elem != endElem; ++elem) {
			auto libmeshElem = Teuchos::rcp_dynamic_cast<
												LibmeshAdapter::LibmeshEntityExtraData<libMesh::Elem>>(
												elem->extraData());
			libMesh::Elem * e = libmeshElem->d_libmesh_geom.get();
			auto nSides = libmeshElem->d_libmesh_geom->n_sides();
			for (auto i = 0; i < nSides; i++) {
				if (boundaryInfo.has_boundary_id(e, i, boundaryId)) {
					local_num_faces++;
				}
			}
		}

	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the data for a boundary.
	 */
	void boundaryDataFunction(std::shared_ptr<void> user_data, const std::string& boundary_name,
			DataTransferKit::View<DataTransferKit::LocalOrdinal> boundary_cells,
			DataTransferKit::View<unsigned> cell_faces_on_boundary) {

		int counter = 0;
		auto boundaryInfo = manager->getMesh()->get_boundary_info();
		auto boundaryId = boundaryInfo.get_id_by_name(boundary_name);

		LibmeshAdapter::ElemPredicateFunction localPredicate =
						[=]( LibmeshAdapter::LibmeshEntity<libMesh::Elem> e) {return e.ownerRank() == thisRank;};


		// Create the entity iterator over those local cells
		auto localElemIter = entitySet->entityIterator(localPredicate);

		// Loop over all nodes and set their spatial coordinates
		unsigned elemCounter = 0;
		auto startElem = localElemIter.begin();
		auto endElem = localElemIter.end();
		for (auto elem = startElem; elem != endElem; ++elem) {
			auto libmeshElem = Teuchos::rcp_dynamic_cast<
												LibmeshAdapter::LibmeshEntityExtraData<libMesh::Elem>>(
												elem->extraData());
			libMesh::Elem * e = libmeshElem->d_libmesh_geom.get();
			auto nSides = libmeshElem->d_libmesh_geom->n_sides();
			for (auto i = 0; i < nSides; i++) {
				if (boundaryInfo.has_boundary_id(e, i, boundaryId)) {
					boundary_cells[counter] = elem->id();
					cell_faces_on_boundary[counter] = i;
					counter++;
				}
			}
		}

	}

//---------------------------------------------------------------------------//
// Degree-of-freedom interface.
//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the size parameters for a degree-of-freedom id map with a single
	 * number of dofs per object.
	 */
	void dofMapSizeFunction(std::shared_ptr<void> user_data,
			size_t &local_num_dofs, size_t &local_num_objects,
			unsigned &dofs_per_object) {

		// local_num_dofs would be total number of degrees of freedom
		// local num objects is number of cells in FEM case (elements)
		// dofs per object is 8 for Hex8
	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the data for a degree-of-freedom id map with a single number of
	 * dofs per object.
	 */
	void dofMapDataFunction(std::shared_ptr<void> user_data,
			DataTransferKit::View<DataTransferKit::GlobalOrdinal> global_dof_ids,
			DataTransferKit::View<DataTransferKit::LocalOrdinal> object_dof_ids,
			std::string &discretization_type) {

	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the size parameters for a degree-of-freedom id map with each
	 * object having a potentially different number of dofs (e.g. mixed topology
	 * cell lists or polyhedron lists).
	 */
	void mixedTopologyDOFMapSizeFunction(std::shared_ptr<void> user_data,
			size_t &local_num_dofs, size_t &local_num_objects,
			size_t &total_dofs_per_object) {

	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the data for a multiple object degree-of-freedom id map
	 * (e.g. mixed topology cell lists or polyhedron lists).
	 */
	void mixedTopologyDOFMapDataFunction(std::shared_ptr<void> user_data,
			DataTransferKit::View<DataTransferKit::GlobalOrdinal> global_dof_ids,
			DataTransferKit::View<DataTransferKit::LocalOrdinal> object_dof_ids,
			DataTransferKit::View<unsigned> dofs_per_object,
			std::string &discretization_type) {

	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Get the size parameters for a field. Field must be of size
	 * local_num_dofs in the associated dof_id_map.
	 */
	template<class Scalar>
	void fieldSizeFunction(std::shared_ptr<void> user_data,
			unsigned &field_dimension, size_t &local_num_dofs) {

		// field dimension is the number of components for all the fields linearlized
		// T is 1, tensor stress can be 9
		// local_num_dofs == same from dof id map function
	}

//---------------------------------------------------------------------------//
	/*!
	 * \brief Pull data from application into a field.
	 */
	template<class Scalar>
	void pullFieldDataFunction(std::shared_ptr<void> user_data,
			DataTransferKit::View<Scalar> field_dofs) {

	}

//---------------------------------------------------------------------------//
	/*
	 * \brief Push data from a field into the application.
	 */
	template<class Scalar>
	void pushFieldDataFunction(std::shared_ptr<void> user_data,
			const DataTransferKit::View<Scalar> field_dofs) {

	}

//---------------------------------------------------------------------------//
	/*
	 * \brief Evaluate a field at a given set of points in a given set of objects.
	 */
	template<class Scalar>
	void evaluateFieldFunction(std::shared_ptr<void> user_data,
			const DataTransferKit::View<DataTransferKit::Coordinate> evaluation_points,
			const DataTransferKit::View<DataTransferKit::LocalOrdinal> object_ids,
			DataTransferKit::View<Scalar> values) {

	}

	/* We don't really need the following functions yet. */
	void boundingVolumeListSize(std::shared_ptr<void> user_data,
			unsigned &space_dim, size_t &local_num_volumes, bool &has_ghosts) {
	}

	void boundingVolumeData(std::shared_ptr<void> user_data,
			DataTransferKit::View<DataTransferKit::Coordinate> bounding_volumes,
			DataTransferKit::View<bool> is_ghost_volume) {

	}
	void polyhedronListSizeFunction(std::shared_ptr<void> user_data,
			unsigned &space_dim, size_t &local_num_nodes,
			size_t &local_num_faces, size_t &total_nodes_per_face,
			size_t &local_num_cells, size_t &total_faces_per_cell,
			bool &has_ghosts) {

	}
	void polyhedronListDataFunction(std::shared_ptr<void> user_data,
			DataTransferKit::View<DataTransferKit::Coordinate> coordinates,
			DataTransferKit::View<DataTransferKit::LocalOrdinal> faces,
			DataTransferKit::View<unsigned> nodes_per_face,
			DataTransferKit::View<DataTransferKit::LocalOrdinal> cells,
			DataTransferKit::View<unsigned> faces_per_cell,
			DataTransferKit::View<int> face_orientation,
			DataTransferKit::View<bool> is_ghost_cell) {

	}
};

}

#endif
