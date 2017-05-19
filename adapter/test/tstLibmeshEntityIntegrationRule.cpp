#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshEntityLocalMapTester
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "LibmeshAdjacencies.hpp"
#include "LibmeshEntity.hpp"
#include "LibmeshEntityIntegrationRule.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_DefaultMpiComm.hpp>

#include <libmesh/cell_hex8.h>
#include <libmesh/equation_systems.h>
#include <libmesh/libmesh.h>
#include <libmesh/linear_implicit_system.h>
#include <libmesh/mesh.h>
#include <libmesh/node.h>
#include <libmesh/parallel.h>
#include <libmesh/point.h>
#include <libmesh/system.h>

//---------------------------------------------------------------------------//
// Hex-8 test.
BOOST_AUTO_TEST_CASE( checkLibmeshEntityIntegrationRule )
{
	const std::string argv_string = "unit_test --keep-cout";
	const char *argv_char = argv_string.c_str();
	Teuchos::GlobalMPISession mpiSession(
			&boost::unit_test::framework::master_test_suite().argc,
			&boost::unit_test::framework::master_test_suite().argv);
	auto comm = Teuchos::DefaultComm<int>::getComm();
	auto mpi_comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(
			comm);
	auto opaque_comm = mpi_comm->getRawMpiComm();
	auto raw_comm = (*opaque_comm)();

    // Create the mesh.
    int space_dim = 3;
    libMesh::LibMeshInit libmesh_init( 1, &argv_char, raw_comm );
    BOOST_VERIFY( libMesh::initialized() );
    BOOST_VERIFY( (int)libmesh_init.comm().rank() == comm->getRank() );
    Teuchos::RCP<libMesh::Mesh> mesh =
        Teuchos::rcp( new libMesh::Mesh( libmesh_init.comm(), space_dim ) );

    // Create the nodes.
    int rank = comm->getRank();
    Teuchos::Array<libMesh::Node *> nodes( 8 );
    double node_coords[3];
    node_coords[0] = 0.0;
    node_coords[1] = 0.0;
    node_coords[2] = 0.0;
    nodes[0] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 0,
        rank );

    node_coords[0] = 1.0;
    node_coords[1] = 0.0;
    node_coords[2] = 0.0;
    nodes[1] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 1,
        rank );

    node_coords[0] = 1.0;
    node_coords[1] = 1.0;
    node_coords[2] = 0.0;
    nodes[2] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 2,
        rank );

    node_coords[0] = 0.0;
    node_coords[1] = 1.0;
    node_coords[2] = 0.0;
    nodes[3] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 3,
        rank );

    node_coords[0] = 0.0;
    node_coords[1] = 0.0;
    node_coords[2] = 1.0;
    nodes[4] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 4,
        rank );

    node_coords[0] = 1.0;
    node_coords[1] = 0.0;
    node_coords[2] = 1.0;
    nodes[5] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 5,
        rank );

    node_coords[0] = 1.0;
    node_coords[1] = 1.0;
    node_coords[2] = 1.0;
    nodes[6] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 6,
        rank );

    node_coords[0] = 0.0;
    node_coords[1] = 1.0;
    node_coords[2] = 1.0;
    nodes[7] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 7,
        rank );

    // Make a hex-8.
    int num_nodes = 8;
    libMesh::Elem *hex_elem = mesh->add_elem( new libMesh::Hex8 );
    hex_elem->processor_id() = rank;
    hex_elem->set_id() = 2 * rank;
    for ( int i = 0; i < num_nodes; ++i )
        hex_elem->set_node( i ) = nodes[i];

    // Check libmesh validity.
    mesh->libmesh_assert_valid_parallel_ids();

    // Make an adjacency data structure.
    LibmeshAdapter::LibmeshAdjacencies adjacencies( mesh );

    // Create a  entity for the hex.
    auto dtk_entity =
        LibmeshAdapter::LibmeshEntity<libMesh::Elem>(
            Teuchos::ptr( hex_elem ), mesh.ptr(),
            Teuchos::ptrFromRef( adjacencies ) );

    // Create an integration rule.
   auto integration_rule =
        Teuchos::rcp( new LibmeshAdapter::LibmeshEntityIntegrationRule() );

    // Test the integration rule.
    Teuchos::Array<Teuchos::Array<double>> p_1;
    Teuchos::Array<double> w_1;
    integration_rule->getIntegrationRule( dtk_entity, 1, p_1, w_1 );
    BOOST_VERIFY( 1 == w_1.size() );
    BOOST_VERIFY( 1 == p_1.size() );
    BOOST_VERIFY( 3 == p_1[0].size() );
    BOOST_VERIFY( 8.0 == w_1[0] );
    BOOST_VERIFY( 0.0 == p_1[0][0] );
    BOOST_VERIFY( 0.0 == p_1[0][1] );
    BOOST_VERIFY( 0.0 == p_1[0][2] );

    Teuchos::Array<Teuchos::Array<double>> p_2;
    Teuchos::Array<double> w_2;
    integration_rule->getIntegrationRule( dtk_entity, 2, p_2, w_2 );
    BOOST_VERIFY( 8 == w_2.size() );
    BOOST_VERIFY( 8 == p_2.size() );
    for ( int i = 0; i < 8; ++i )
    {
    	BOOST_VERIFY( w_2[i] == 1.0 );
    	BOOST_VERIFY( p_2[i].size() == 3 );

        for ( int d = 0; d < 3; ++d )
        {
			BOOST_TEST(std::abs( p_2[i][d] ) == ( 1.0 / std::sqrt( 3.0 )),
					boost::test_tools::tolerance(1.0e-15));
		}
    }
}

//---------------------------------------------------------------------------//
// end of tstLibmeshEntityIntegrationRule.cpp
//---------------------------------------------------------------------------//
