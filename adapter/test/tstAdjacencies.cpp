#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshEntityAdjacenciesTester
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include <LibmeshAdjacencies.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Tuple.hpp>

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
BOOST_AUTO_TEST_CASE( checkLibmeshAdjacencies )
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

//    // Extract the raw mpi communicator.
//    Teuchos::RCP<const Teuchos::Comm<int>> comm = getDefaultComm<int>();
//    Teuchos::RCP<const Teuchos::MpiComm<int>> mpi_comm =
//        Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>( comm );
//    Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm>> opaque_comm =
//        mpi_comm->getRawMpiComm();
//    MPI_Comm raw_comm = ( *opaque_comm )();

    // Create the mesh.
    int space_dim = 3;
//    argv_string = "--keep-cout";
//    argv_char = argv_string.c_str();
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
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ),
        8 * rank + 0, rank );

    node_coords[0] = 1.0;
    node_coords[1] = 0.0;
    node_coords[2] = 0.0;
    nodes[1] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ),
        8 * rank + 1, rank );

    node_coords[0] = 1.0;
    node_coords[1] = 1.0;
    node_coords[2] = 0.0;
    nodes[2] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ),
        8 * rank + 2, rank );

    node_coords[0] = 0.0;
    node_coords[1] = 1.0;
    node_coords[2] = 0.0;
    nodes[3] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ),
        8 * rank + 3, rank );

    node_coords[0] = 0.0;
    node_coords[1] = 0.0;
    node_coords[2] = 1.0;
    nodes[4] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ),
        8 * rank + 4, rank );

    node_coords[0] = 1.0;
    node_coords[1] = 0.0;
    node_coords[2] = 1.0;
    nodes[5] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ),
        8 * rank + 5, rank );

    node_coords[0] = 1.0;
    node_coords[1] = 1.0;
    node_coords[2] = 1.0;
    nodes[6] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ),
        8 * rank + 6, rank );

    node_coords[0] = 0.0;
    node_coords[1] = 1.0;
    node_coords[2] = 1.0;
    nodes[7] = mesh->add_point(
        libMesh::Point( node_coords[0], node_coords[1], node_coords[2] ),
        8 * rank + 7, rank );

    // Make a hex-8.
    libMesh::Elem *hex_elem_1 = mesh->add_elem( new libMesh::Hex8 );
    hex_elem_1->processor_id() = rank;
    hex_elem_1->set_id() = 2 * rank;
    for ( int i = 0; i < 8; ++i )
        hex_elem_1->set_node( i ) = nodes[i];

    // Make another hex-8.
    libMesh::Elem *hex_elem_2 = mesh->add_elem( new libMesh::Hex8 );
    hex_elem_2->processor_id() = rank;
    hex_elem_2->set_id() = 2 * rank + 1;
    for ( int i = 0; i < 8; ++i )
        hex_elem_2->set_node( i ) = nodes[i];

    // Check libmesh validity.
    mesh->libmesh_assert_valid_parallel_ids();

    // Make an adjacency data structure.
    LibmeshAdapter::LibmeshAdjacencies adjacencies( mesh );

    // Check the node adjacencies of the first hex elem.
    unsigned int num_nodes = 8;
    Teuchos::Array<Teuchos::Ptr<libMesh::Node>> elem_1_nodes;
    adjacencies.getLibmeshAdjacencies( Teuchos::ptr( hex_elem_1 ),
                                       elem_1_nodes );
    BOOST_VERIFY( num_nodes == elem_1_nodes.size() );
    for ( unsigned int n = 0; n < num_nodes; ++n )
    {
    	BOOST_VERIFY( nodes[n]->id() == elem_1_nodes[n]->id() );
    };

    // Check the node adjacencies of the second hex elem.
    Teuchos::Array<Teuchos::Ptr<libMesh::Node>> elem_2_nodes;
    adjacencies.getLibmeshAdjacencies( Teuchos::ptr( hex_elem_2 ),
                                       elem_2_nodes );
    BOOST_VERIFY( num_nodes == elem_2_nodes.size() );
    for ( unsigned int n = 0; n < num_nodes; ++n )
    {
    	BOOST_VERIFY( nodes[n]->id() == elem_2_nodes[n]->id() );
    };

    // Check the elem adjacencies of the first hex.
    Teuchos::Array<Teuchos::Ptr<libMesh::Elem>> elem_1_elems;
    adjacencies.getLibmeshAdjacencies( Teuchos::ptr( hex_elem_1 ),
                                       elem_1_elems );
    BOOST_VERIFY( 0 == elem_1_elems.size() );

    // Check the elem adjacencies of the second hex.
    Teuchos::Array<Teuchos::Ptr<libMesh::Elem>> elem_2_elems;
    adjacencies.getLibmeshAdjacencies( Teuchos::ptr( hex_elem_2 ),
                                       elem_2_elems );
    BOOST_VERIFY( 0 == elem_2_elems.size() );

    // Check the adjacencies of the nodes.
    for ( unsigned int n = 0; n < num_nodes; ++n )
    {
        Teuchos::Array<Teuchos::Ptr<libMesh::Elem>> node_elems;
        adjacencies.getLibmeshAdjacencies( Teuchos::ptr( nodes[n] ),
                                           node_elems );
        BOOST_VERIFY( 2 == node_elems.size() );
        BOOST_VERIFY( hex_elem_2->id() == node_elems[0]->id() );
        BOOST_VERIFY( hex_elem_1->id() == node_elems[1]->id() );

        Teuchos::Array<Teuchos::Ptr<libMesh::Node>> node_nodes;
        adjacencies.getLibmeshAdjacencies( Teuchos::ptr( nodes[n] ),
                                           node_nodes );
        BOOST_VERIFY( 0 == node_nodes.size() );
    }
}

//---------------------------------------------------------------------------//
// end tstLibmeshAdjacencies.cpp
//---------------------------------------------------------------------------//
