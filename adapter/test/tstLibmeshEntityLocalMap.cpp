#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshEntityLocalMapTester
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include <LibmeshEntity.hpp>
#include <LibmeshEntityExtraData.hpp>
#include <LibmeshEntityLocalMap.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_TypeTraits.hpp>

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
// MPI Setup
//---------------------------------------------------------------------------//

template <class Ordinal>
Teuchos::RCP<const Teuchos::Comm<Ordinal>> getDefaultComm()
{
#ifdef HAVE_MPI
    return Teuchos::DefaultComm<Ordinal>::getComm();
#else
    return Teuchos::rcp( new Teuchos::SerialComm<Ordinal>() );
#endif
}

//---------------------------------------------------------------------------//
// TEST EPSILON
//---------------------------------------------------------------------------//

const double epsilon = 1.0e-14;

//---------------------------------------------------------------------------//
// Hex-8 test.
BOOST_AUTO_TEST_CASE( checkEntityLocalMap )
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
    node_coords[2] = -2.0;
    nodes[0] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 0,
        rank );

    node_coords[0] = 2.0;
    node_coords[1] = 0.0;
    node_coords[2] = -2.0;
    nodes[1] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 1,
        rank );

    node_coords[0] = 2.0;
    node_coords[1] = 2.0;
    node_coords[2] = -2.0;
    nodes[2] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 2,
        rank );

    node_coords[0] = 0.0;
    node_coords[1] = 2.0;
    node_coords[2] = -2.0;
    nodes[3] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 3,
        rank );

    node_coords[0] = 0.0;
    node_coords[1] = 0.0;
    node_coords[2] = 0.0;
    nodes[4] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 4,
        rank );

    node_coords[0] = 2.0;
    node_coords[1] = 0.0;
    node_coords[2] = 0.0;
    nodes[5] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 5,
        rank );

    node_coords[0] = 2.0;
    node_coords[1] = 2.0;
    node_coords[2] = 0.0;
    nodes[6] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 6,
        rank );

    node_coords[0] = 0.0;
    node_coords[1] = 2.0;
    node_coords[2] = 0.0;
    nodes[7] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ), 7,
        rank );

    // Make a hex-8.
    libMesh::Elem *hex_elem = mesh->add_elem( new libMesh::Hex8 );
    hex_elem->processor_id() = rank;
    hex_elem->set_id() = 2 * rank;
    for ( int i = 0; i < 8; ++i )
        hex_elem->set_node( i ) = nodes[i];

    // Check libmesh validity.
    mesh->libmesh_assert_valid_parallel_ids();

    // Make an adjacency data structure.
    LibmeshAdapter::LibmeshAdjacencies adjacencies( mesh );

    // Create a  entity for the hex.
    LibmeshAdapter::LibmeshEntity<libMesh::Elem> dtk_entity =
        LibmeshAdapter::LibmeshEntity<libMesh::Elem>(
            Teuchos::ptr( hex_elem ), mesh.ptr(),
            Teuchos::ptrFromRef( adjacencies ) );

    // Make a libmesh system. We will put a first order linear basis on the
    // elements.
    libMesh::EquationSystems equation_systems( *mesh );
    libMesh::LinearImplicitSystem &system =
        equation_systems.add_system<libMesh::LinearImplicitSystem>( "Test" );
    system.add_variable( "test_var", libMesh::FIRST );

    // Create a local map from the libmesh mesh.
    auto local_map =
        Teuchos::rcp( new LibmeshAdapter::LibmeshEntityLocalMap(
            mesh, Teuchos::rcpFromRef( system ) ) );

    // Test the measure.
    BOOST_TEST( local_map->measure( dtk_entity ) == 8.0, boost::test_tools::tolerance(epsilon) );

    // Test the centroid.
    Teuchos::Array<double> centroid( space_dim, 0.0 );
    local_map->centroid( dtk_entity, centroid() );
    BOOST_VERIFY( centroid[0] == 1.0 );
    BOOST_VERIFY( centroid[1] ==  1.0 );
    BOOST_VERIFY( centroid[2] == -1.0 );

    // Make a good point and a bad point.
    Teuchos::Array<double> good_point( space_dim );
    good_point[0] = 0.5;
    good_point[1] = 1.5;
    good_point[2] = -1.0;
    Teuchos::Array<double> bad_point( space_dim );
    bad_point[0] = 0.75;
    bad_point[1] = -1.75;
    bad_point[2] = 0.35;

    // Test the reference frame safeguard.
    BOOST_VERIFY(
        local_map->isSafeToMapToReferenceFrame( dtk_entity, good_point() ) );
    BOOST_VERIFY(
        !local_map->isSafeToMapToReferenceFrame( dtk_entity, bad_point() ) );

    // Test the mapping to reference frame.
    Teuchos::Array<double> ref_good_point( space_dim );
    bool good_map = local_map->mapToReferenceFrame( dtk_entity, good_point(),
                                                    ref_good_point() );
    BOOST_VERIFY( good_map );
    BOOST_TEST( ref_good_point[0] == -0.5, boost::test_tools::tolerance(epsilon) );
    BOOST_TEST( ref_good_point[1] == 0.5, boost::test_tools::tolerance(epsilon) );
    BOOST_VERIFY( std::abs( ref_good_point[2] ) < epsilon );

    Teuchos::Array<double> ref_bad_point( space_dim );
    local_map->mapToReferenceFrame( dtk_entity, bad_point(), ref_bad_point() );

    // Test the point inclusion.
    BOOST_VERIFY(
        local_map->checkPointInclusion( dtk_entity, ref_good_point() ) );
    BOOST_VERIFY(
        !local_map->checkPointInclusion( dtk_entity, ref_bad_point() ) );

    // Test the map to physical frame.
    Teuchos::Array<double> phy_good_point( space_dim );
    local_map->mapToPhysicalFrame( dtk_entity, ref_good_point(),
                                   phy_good_point() );
    BOOST_TEST( good_point[0] == phy_good_point[0], boost::test_tools::tolerance(epsilon) );
    BOOST_TEST( good_point[1] == phy_good_point[1], boost::test_tools::tolerance(epsilon) );
    BOOST_TEST( good_point[2] == phy_good_point[2], boost::test_tools::tolerance(epsilon) );

    Teuchos::Array<double> phy_bad_point( space_dim );
    local_map->mapToPhysicalFrame( dtk_entity, ref_bad_point(),
                                   phy_bad_point() );
    BOOST_TEST( bad_point[0] == phy_bad_point[0], boost::test_tools::tolerance(epsilon) );
    BOOST_TEST( bad_point[1] == phy_bad_point[1], boost::test_tools::tolerance(epsilon) );
    BOOST_TEST( bad_point[2] == phy_bad_point[2], boost::test_tools::tolerance(epsilon) );

    // Test the coordinates of the points extracted through the centroid
    // function.
    Teuchos::Array<double> point_coords( space_dim );
    int num_nodes = 8;
    for ( int n = 0; n < num_nodes; ++n )
    {
        auto dtk_node = LibmeshAdapter::LibmeshEntity<libMesh::Node>(
            Teuchos::ptr( nodes[n] ), mesh.ptr(),
            Teuchos::ptrFromRef( adjacencies ) );
        local_map->centroid( dtk_node, point_coords() );
        BOOST_TEST( ( *nodes[n] )( 0 ) == point_coords[0] );
        BOOST_TEST( ( *nodes[n] )( 1 ) == point_coords[1] );
        BOOST_TEST( ( *nodes[n] )( 2 ) == point_coords[2] );
    }
}

//---------------------------------------------------------------------------//
// end tstLibmeshEntityLocalMap.cpp
//---------------------------------------------------------------------------//
