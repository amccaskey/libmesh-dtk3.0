#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshUserAppTester

#include <boost/test/included/unit_test.hpp>
#include "Teuchos_OpaqueWrapper.hpp"
#include <LibmeshUserApplication.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

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

		// Create a vector from the variable over all subdomains.
		libmeshManager = std::make_shared<LibmeshAdapter::LibmeshManager>(
				Teuchos::rcp(mesh.get(), false), Teuchos::rcpFromRef(system));

		// Set the user functions.
		registry = std::make_shared<
				DataTransferKit::UserFunctionRegistry<double>>();
		registry->setNodeListSizeFunction(LibmeshApp::nodeListSize,
				libmeshManager);
		registry->setNodeListDataFunction(LibmeshApp::nodeListData,
				libmeshManager);

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

	// We know the coords are as follows:
	// FIXME AUTO GENERATE BASED ON MESH
	std::vector<double> expectedCoords { 0, 0, 0, 0.25, 0, 0, 0.25, 0.25, 0, 0,
			0.25, 0, 0, 0, 0.25, 0.25, 0, 0.25, 0.25, 0.25, 0.25, 0, 0.25, 0.25,
			0.5, 0, 0, 0.5, 0.25, 0, 0.5, 0, 0.25, 0.5, 0.25, 0.25, 0.75, 0, 0,
			0.75, 0.25, 0, 0.75, 0, 0.25, 0.75, 0.25, 0.25, 1, 0, 0, 1, 0.25, 0,
			1, 0, 0.25, 1, 0.25, 0.25, 0.25, 0.5, 0, 0, 0.5, 0, 0.25, 0.5, 0.25,
			0, 0.5, 0.25, 0.5, 0.5, 0, 0.5, 0.5, 0.25, 0.75, 0.5, 0, 0.75, 0.5,
			0.25, 1, 0.5, 0, 1, 0.5, 0.25, 0.25, 0.75, 0, 0, 0.75, 0, 0.25,
			0.75, 0.25, 0, 0.75, 0.25, 0.5, 0.75, 0, 0.5, 0.75, 0.25, 0.75,
			0.75, 0, 0.75, 0.75, 0.25, 1, 0.75, 0, 1, 0.75, 0.25, 0.25, 1, 0, 0,
			1, 0, 0.25, 1, 0.25, 0, 1, 0.25, 0.5, 1, 0, 0.5, 1, 0.25, 0.75, 1,
			0, 0.75, 1, 0.25, 1, 1, 0, 1, 1, 0.25, 0, 0, 0.5, 0.25, 0, 0.5,
			0.25, 0.25, 0.5, 0, 0.25, 0.5, 0.5, 0, 0.5, 0.5, 0.25, 0.5, 0.75, 0,
			0.5, 0.75, 0.25, 0.5, 1, 0, 0.5, 1, 0.25, 0.5, 0.25, 0.5, 0.5, 0,
			0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.5, 0.5, 1, 0.5, 0.5, 0.25, 0.75,
			0.5, 0, 0.75, 0.5, 0.5, 0.75, 0.5, 0.75, 0.75, 0.5, 1, 0.75, 0.5,
			0.25, 1, 0.5, 0, 1, 0.5, 0.5, 1, 0.5, 0.75, 1, 0.5, 1, 1, 0.5, 0, 0,
			0.75, 0.25, 0, 0.75, 0.25, 0.25, 0.75, 0, 0.25, 0.75, 0.5, 0, 0.75,
			0.5, 0.25, 0.75, 0.75, 0, 0.75, 0.75, 0.25, 0.75, 1, 0, 0.75, 1,
			0.25, 0.75, 0.25, 0.5, 0.75, 0, 0.5, 0.75, 0.5, 0.5, 0.75, 0.75,
			0.5, 0.75, 1, 0.5, 0.75, 0.25, 0.75, 0.75, 0, 0.75, 0.75, 0.5, 0.75,
			0.75, 0.75, 0.75, 0.75, 1, 0.75, 0.75, 0.25, 1, 0.75, 0, 1, 0.75,
			0.5, 1, 0.75, 0.75, 1, 0.75, 1, 1, 0.75, 0, 0, 1, 0.25, 0, 1, 0.25,
			0.25, 1, 0, 0.25, 1, 0.5, 0, 1, 0.5, 0.25, 1, 0.75, 0, 1, 0.75,
			0.25, 1, 1, 0, 1, 1, 0.25, 1, 0.25, 0.5, 1, 0, 0.5, 1, 0.5, 0.5, 1,
			0.75, 0.5, 1, 1, 0.5, 1, 0.25, 0.75, 1, 0, 0.75, 1, 0.5, 0.75, 1,
			0.75, 0.75, 1, 1, 0.75, 1, 0.25, 1, 1, 0, 1, 1, 0.5, 1, 1, 0.75, 1,
			1, 1, 1, 1 };

	// Verify the number of Nodes
	auto node_list = user_app->getNodeList();
	BOOST_VERIFY(node_list.coordinates.size() == 375);
	BOOST_VERIFY((node_list.coordinates.size() / expectedDim) == expectednNodes);

	unsigned counter = 0;
	for (unsigned i = 0; i < expectednNodes; ++i) {
		for (unsigned d = 0; d < expectedDim; ++d) {
			BOOST_VERIFY(
					node_list.coordinates(i, d) == expectedCoords[counter]);
			counter++;
		}
	}
}
