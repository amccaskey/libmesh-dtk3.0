#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshEntityTester
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include <LibmeshAdjacencies.hpp>
#include <LibmeshEntity.hpp>
#include <LibmeshEntityExtraData.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <libmesh/cell_hex8.h>
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/node.h>
#include <libmesh/parallel.h>
#include <libmesh/point.h>

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
// Hex-8 test.
BOOST_AUTO_TEST_CASE( checkLibmeshEntity )
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
    libMesh::Elem *hex_elem = mesh->add_elem( new libMesh::Hex8 );
    hex_elem->processor_id() = rank;
    hex_elem->set_id() = 2 * rank;
    for ( int i = 0; i < 8; ++i )
        hex_elem->set_node( i ) = nodes[i];

    // Make 2 subdomains and put the hex-8 in the first subdomain.
    int subdomain_1_id = 1;
    int subdomain_2_id = 2;
    std::set<libMesh::subdomain_id_type> subdomain_ids;
    subdomain_ids.insert( subdomain_1_id );
    subdomain_ids.insert( subdomain_2_id );
    hex_elem->subdomain_id() = subdomain_1_id;

    // Make 2 boundaries and add the first elem side to one and first node to
    // the second.
    int boundary_1_id = 1;
    int boundary_2_id = 2;
    mesh->get_boundary_info().add_side( hex_elem, 0, boundary_1_id );
    mesh->get_boundary_info().add_node( nodes[0], boundary_2_id );

    // Check libmesh validity.
    mesh->libmesh_assert_valid_parallel_ids();

    // Make an adjacency data structure.
    LibmeshAdapter::LibmeshAdjacencies adjacencies( mesh );

    // Create a  entity for the hex.
   auto dtk_entity =
        LibmeshAdapter::LibmeshEntity<libMesh::Elem>(
            Teuchos::ptr( hex_elem ), mesh.ptr(),
            Teuchos::ptrFromRef( adjacencies ) );

    // Print out the entity.
    Teuchos::RCP<Teuchos::FancyOStream> fancy_out =
        Teuchos::VerboseObjectBase::getDefaultOStream();
//    dtk_entity.describe( *fancy_out );

    // Test the entity.
    BOOST_VERIFY( hex_elem->id() == dtk_entity.id() );
    BOOST_VERIFY( rank == dtk_entity.ownerRank() );
    BOOST_VERIFY( space_dim == dtk_entity.topologicalDimension() );
    BOOST_VERIFY( space_dim == dtk_entity.physicalDimension() );

    BOOST_VERIFY( dtk_entity.inBlock( subdomain_1_id ) );
    BOOST_VERIFY( !dtk_entity.inBlock( subdomain_2_id ) );

    BOOST_VERIFY( dtk_entity.onBoundary( boundary_1_id ) );
    BOOST_VERIFY( !dtk_entity.onBoundary( boundary_2_id ) );

   auto elem_extra_data =
        dtk_entity.extraData();
   BOOST_VERIFY( hex_elem ==
                   Teuchos::rcp_dynamic_cast<
                       LibmeshAdapter::LibmeshEntityExtraData<libMesh::Elem>>(
                       elem_extra_data )
                       ->d_libmesh_geom.getRawPtr() );

    Teuchos::Tuple<double, 6> hex_bounds;
    dtk_entity.boundingBox( hex_bounds );
    BOOST_VERIFY( 0.0 == hex_bounds[0] );
    BOOST_VERIFY( 0.0 == hex_bounds[1] );
    BOOST_VERIFY( 0.0 == hex_bounds[2] );
    BOOST_VERIFY( 1.0 == hex_bounds[3] );
    BOOST_VERIFY( 1.0 == hex_bounds[4] );
    BOOST_VERIFY( 1.0 == hex_bounds[5] );

    // Test a node.
  auto dtk_node =
        LibmeshAdapter::LibmeshEntity<libMesh::Node>(
            Teuchos::ptr( nodes[0] ), mesh.ptr(),
            Teuchos::ptrFromRef( adjacencies ) );
  BOOST_VERIFY( nodes[0]->id() == dtk_node.id() );
  BOOST_VERIFY( rank == dtk_node.ownerRank() );
  BOOST_VERIFY( 0 == dtk_node.topologicalDimension() );
  BOOST_VERIFY( space_dim == dtk_node.physicalDimension() );

  BOOST_VERIFY( dtk_node.inBlock( subdomain_1_id ) );
  BOOST_VERIFY( !dtk_node.inBlock( subdomain_2_id ) );

  BOOST_VERIFY( !dtk_node.onBoundary( boundary_1_id ) );
  BOOST_VERIFY( dtk_node.onBoundary( boundary_2_id ) );

    auto node_extra_data =
        dtk_node.extraData();
    BOOST_VERIFY( nodes[0] ==
                   Teuchos::rcp_dynamic_cast<
				   LibmeshAdapter::LibmeshEntityExtraData<libMesh::Node>>(
                       node_extra_data )
                       ->d_libmesh_geom.getRawPtr() );

    Teuchos::Tuple<double, 6> node_bounds;
    dtk_node.boundingBox( node_bounds );
    BOOST_VERIFY( 0.0 == node_bounds[0] );
    BOOST_VERIFY( 0.0 == node_bounds[1] );
    BOOST_VERIFY( 0.0 == node_bounds[2] );
    BOOST_VERIFY( 0.0 == node_bounds[3] );
    BOOST_VERIFY( 0.0 == node_bounds[4] );
    BOOST_VERIFY( 0.0 == node_bounds[5] );

    // Print out the node.
//    dtk_node.describe( *fancy_out );
}

//---------------------------------------------------------------------------//
// end tstLibmeshEntity.cpp
//---------------------------------------------------------------------------//
