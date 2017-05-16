
#include <boost/test/included/unit_test.hpp>

#include <DTK_UserApplication.hpp>
#include <DTK_UserDataInterface.hpp>
#include <DTK_UserFunctionRegistry.hpp>
#include <DTK_View.hpp>

#include <DTK_LibmeshManager.hpp>
#include <DTK_LibmeshEntity.hpp>
#include <DTK_LibmeshEntityExtraData.hpp>
#include <DTK_LibmeshEntityIterator.hpp>
#include <DTK_BasicEntityPredicates.hpp>

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
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include <memory>


void nodeListSize(std::shared_ptr<void> user_data, unsigned &space_dim,
		        size_t &local_num_nodes, bool &has_ghosts) {

	    auto u = std::static_pointer_cast<DataTransferKit::LibmeshManager>(user_data);

	        DataTransferKit::LocalEntityPredicate localPredicate(
				            u->entitySet()->communicator()->getRank());

		    auto entitySet = u->entitySet();

		        auto localNodeIter = entitySet->entityIterator(0, localPredicate);
			    auto totalNodeIter = entitySet->entityIterator(0);

			        local_num_nodes = localNodeIter.size();
				    space_dim = entitySet->physicalDimension();
				        has_ghosts = totalNodeIter.size() != local_num_nodes;
}


