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

void pullFieldData( std::shared_ptr<void> user_data,
                    DataTransferKit::View<double> field_dofs )
{
    auto u = std::static_pointer_cast<LibmeshAdapter::LibmeshManager>(
        user_data );

	// Get the EntitySet
	auto entitySet = u->entitySet();

	// Get the spacial dimenstion
	auto space_dim = entitySet->physicalDimension();
	// Create a predicate that picks out only local nodes
	auto thisRank = u->entitySet()->communicator()->getRank();
	LibmeshAdapter::NodePredicateFunction localPredicate =
			[=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> e) {return e.ownerRank() == thisRank;};

	// Create the entity iterator over those local nodes
	auto localNodeIter = entitySet->entityIterator(localPredicate);

	// Loop over all nodes and set their spatial coordinates
	unsigned num_nodes = localNodeIter.size();

}

}

#endif
