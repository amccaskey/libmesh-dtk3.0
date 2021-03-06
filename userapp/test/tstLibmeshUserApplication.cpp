#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshUserAppTester

#include <boost/test/included/unit_test.hpp>
#include "Teuchos_OpaqueWrapper.hpp"
#include <LibmeshUserApplication.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/analytic_function.h"

void exact_solution_wrapper (libMesh::DenseVector<libMesh::Number> & output,
                             const libMesh::Point & p,
                             const libMesh::Real)
{
  output(0) = 0.0;
}

struct TestFixture {
	static TestFixture*& instance() {
		static TestFixture* inst = 0;
		return inst;
	}

	~TestFixture() {
		BOOST_TEST_MESSAGE("teardown fixture");
	}

	std::shared_ptr<boost::mpi::environment> env;
	std::shared_ptr<boost::mpi::communicator> comm;

	TestFixture() :
			env(std::make_shared<boost::mpi::environment>()), comm(
					std::make_shared<boost::mpi::communicator>()) {
		BOOST_TEST_MESSAGE("setup fixture");
		instance() = this;

		// Create the mesh.
		BOOST_VERIFY(!libMesh::initialized());
		libmesh_init = std::make_shared<libMesh::LibMeshInit>(
				boost::unit_test::framework::master_test_suite().argc,
				boost::unit_test::framework::master_test_suite().argv,
				*comm.get());
		BOOST_VERIFY(libMesh::initialized());
		BOOST_VERIFY((int )libmesh_init->comm().rank() == comm->rank());
		mesh = std::make_shared<libMesh::Mesh>(libmesh_init->comm());
		libMesh::MeshTools::Generation::build_cube(*mesh.get(), 4, 4, 4, 0.0,
				1.0, 0.0, 1.0, 0.0, 1.0, libMesh::HEX8);

		// Make a libmesh system. We will put a first order linear basis on the
		// elements for all subdomains.
		std::string var_name = "test_var";
		libMesh::EquationSystems equation_systems(*mesh.get());
		libMesh::ExplicitSystem &system = equation_systems.add_system<
				libMesh::ExplicitSystem>("Test System");
		int var_id = system.add_variable(var_name);
		equation_systems.init();

		// Put some data in the variable.
		int sys_id = system.number();
		libMesh::MeshBase::node_iterator nodes_begin =
				mesh->local_nodes_begin();
		libMesh::MeshBase::node_iterator nodes_end = mesh->local_nodes_end();
		for (auto node_it = nodes_begin; node_it != nodes_end; ++node_it) {
			auto dof_id = (*node_it)->dof_number(sys_id, var_id, 0);
			system.solution->set(dof_id, (*node_it)->id());
		}
		system.solution->close();
		system.update();

		std::set<libMesh::boundary_id_type> boundary_ids;
		// the dim==1 mesh has two boundaries with IDs 0 and 1
		boundary_ids.insert(0);
		boundary_ids.insert(1);
		boundary_ids.insert(2);
		boundary_ids.insert(3);
		boundary_ids.insert(4);
		boundary_ids.insert(5);

		std::vector<unsigned int> variables(1);
		variables[0] = var_id;

		libMesh::AnalyticFunction<> exact_solution_object(exact_solution_wrapper);

		libMesh::DirichletBoundary dirichlet_bc(boundary_ids, variables, exact_solution_object);

//		system.get_dof_map().add_dirichlet_boundary(dirichlet_bc);

		equation_systems.init();

//		std::cout << "HELLO MESH:\n" << mesh->get_info() << "\n";
//		mesh->get_boundary_info().print_info(std::cout);
		// Create a vector from the variable over all subdomains.
		libmeshManager = std::make_shared<LibmeshAdapter::LibmeshManager>(
				Teuchos::rcp(mesh.get(), false), Teuchos::rcpFromRef(system));

		// Set the user functions.
		LibmeshApp::LibmeshUserApplication app(libmeshManager);
		auto nodeListSizeFunc = std::bind(
				&LibmeshApp::LibmeshUserApplication::nodeListSize, app,
				std::placeholders::_1, std::placeholders::_2,
				std::placeholders::_3, std::placeholders::_4);
		auto nodeListDataFunc = std::bind(
				&LibmeshApp::LibmeshUserApplication::nodeListData, app,
				std::placeholders::_1, std::placeholders::_2,
				std::placeholders::_3);

		auto cellListSizeFunc = std::bind(
				&LibmeshApp::LibmeshUserApplication::cellListSize, app,
				std::placeholders::_1, std::placeholders::_2,
				std::placeholders::_3, std::placeholders::_4,
				std::placeholders::_5, std::placeholders::_6);
		auto cellListDataFunc = std::bind(
				&LibmeshApp::LibmeshUserApplication::cellListData, app,
				std::placeholders::_1, std::placeholders::_2,
				std::placeholders::_3, std::placeholders::_4,
				std::placeholders::_5);

		auto boundarySizeFunc = std::bind(
				&LibmeshApp::LibmeshUserApplication::boundarySizeFunction, app,
				std::placeholders::_1, std::placeholders::_2,
				std::placeholders::_3);
		auto boundaryDataFunc = std::bind(
				&LibmeshApp::LibmeshUserApplication::boundaryDataFunction, app,
				std::placeholders::_1, std::placeholders::_2,
				std::placeholders::_3,  std::placeholders::_4);

		registry = std::make_shared<
				DataTransferKit::UserFunctionRegistry<double>>();
		registry->setNodeListSizeFunction(nodeListSizeFunc,
				libmeshManager);
		registry->setNodeListDataFunction(nodeListDataFunc,
				libmeshManager);

		registry->setCellListSizeFunction(cellListSizeFunc, libmeshManager);
		registry->setCellListDataFunction(cellListDataFunc, libmeshManager);

		registry->setBoundarySizeFunction(boundarySizeFunc, libmeshManager);
		registry->setBoundaryDataFunction(boundaryDataFunc, libmeshManager);

		BOOST_VERIFY(mesh);
		BOOST_VERIFY(mesh->n_local_nodes() == 125);
	}

	std::shared_ptr<LibmeshAdapter::LibmeshManager> libmeshManager;
	std::shared_ptr<libMesh::Mesh> mesh;
	std::shared_ptr<DataTransferKit::UserFunctionRegistry<double>> registry;
	std::shared_ptr<libMesh::LibMeshInit> libmesh_init;
};

BOOST_GLOBAL_FIXTURE(TestFixture);

BOOST_AUTO_TEST_CASE(checkNodeList) {
	auto fixture = TestFixture::instance();
	auto libmeshManager = fixture->libmeshManager;

	// Get a node list.
	// Create the user application.
	auto user_app = std::make_shared<
			DataTransferKit::UserApplication<double,
					DataTransferKit::Serial>>(fixture->registry);

	// We know this mesh has dim = 3, and nNodes = 125
	auto expectedDim = 3;
	auto expectednNodes = 125;

	// Create a predicate that picks out only local nodes
	auto entitySet = libmeshManager->entitySet();
	auto thisRank = entitySet->communicator()->getRank();
	LibmeshAdapter::NodePredicateFunction localPredicate =
			[=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> e) {return e.ownerRank() == thisRank;};

	// Create the entity iterator over those local nodes
	auto localNodeIter = entitySet->entityIterator(localPredicate);

	// Verify the number of Nodes
	auto node_list = user_app->getNodeList();
	BOOST_VERIFY(node_list.coordinates.size() == 375);
	BOOST_VERIFY((node_list.coordinates.size() / expectedDim) == expectednNodes);

	// Loop over all nodes and set their spatial coordinates
	unsigned num_nodes = localNodeIter.size();
	unsigned counter = 0;
	auto startNode = localNodeIter.begin();
	auto endNode = localNodeIter.end();
	for (auto node = startNode; node != endNode; ++node) {
		auto libmeshNode = Teuchos::rcp_dynamic_cast<
				LibmeshAdapter::LibmeshEntityExtraData<libMesh::Node>>(
				node->extraData());
		for (unsigned d = 0; d < expectedDim; ++d) {
			BOOST_VERIFY(
					node_list.coordinates(counter, d)
							== libmeshNode->d_libmesh_geom->operator()(d));
		}
		counter++;
	}
}

BOOST_AUTO_TEST_CASE(checkCellList) {
	auto fixture = TestFixture::instance();
	auto libmeshManager = fixture->libmeshManager;

	// Get a node list.
	// Create the user application.
	auto user_app = std::make_shared<
			DataTransferKit::UserApplication<double,
					DataTransferKit::Serial>>(fixture->registry);


	std::vector<std::string> cell_topologies {"HEX8"};
	auto cell_list = user_app->getCellList(cell_topologies);

	BOOST_VERIFY(cell_list.cells.rank() == 2);
	BOOST_VERIFY(cell_list.cells.size() == 8 * 64);
	BOOST_VERIFY(cell_list.cells.dimension(0) == 64);
	BOOST_VERIFY(cell_list.cells.dimension(1) == 8);

	// Create a predicate that picks out only local cells
	auto entitySet = libmeshManager->entitySet();

	auto thisRank = entitySet->communicator()->getRank();
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
		for (int i = 0; i < 8; i++) {
			BOOST_VERIFY(cell_list.cells[elemCounter*8 + i] == libmeshElem->d_libmesh_geom->node(i));
		}

		elemCounter++;
	}

	BOOST_VERIFY(cell_list.cells[511] == 123);
	BOOST_VERIFY(cell_list.cells(63, 7) == 123);
}


BOOST_AUTO_TEST_CASE(checkBoundaries) {
	auto fixture = TestFixture::instance();
	auto libmeshManager = fixture->libmeshManager;

	// Get a node list.
	// Create the user application.
	auto user_app = std::make_shared<
			DataTransferKit::UserApplication<double,
					DataTransferKit::Serial>>(fixture->registry);
	std::vector<std::string> cell_topologies {"HEX8"};
	auto cell_list = user_app->getCellList(cell_topologies);
	user_app->getBoundary("bottom", cell_list);

	auto boundaryCells = cell_list.boundary_cells;


	// THis is a 4x4x4 mesh, so the bottom boundary
	// condition should have faces from  16 cells that
	// make it up
	BOOST_VERIFY(boundaryCells.size() == 16);



}
