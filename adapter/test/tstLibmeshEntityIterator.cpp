#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshEntityIteratorTester
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
#include <LibmeshEntityIterator.hpp>

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
BOOST_AUTO_TEST_CASE( checkLibmeshEntityIterator )
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

    // Make an iterator for the hex.
    LibmeshAdapter::ElemPredicateFunction all_pred =
        [=]( LibmeshAdapter::LibmeshEntity<libMesh::Elem> ) { return true; };
    auto entity_iterator =
    		LibmeshAdapter::LibmeshEntityIterator<
            libMesh::Mesh::const_element_iterator>(
            mesh->elements_begin(), mesh->elements_begin(),
            mesh->elements_end(), mesh.ptr(),
            Teuchos::ptrFromRef( adjacencies ), all_pred );

    // Test the entity iterator.
    unsigned int num_hex = 1;
    BOOST_VERIFY( entity_iterator.size() == num_hex );
    BOOST_VERIFY( entity_iterator == entity_iterator.begin() );
    BOOST_VERIFY( entity_iterator != entity_iterator.end() );

    // Test the first entity under the iterator with a pointer dereference.
    BOOST_VERIFY( hex_elem->id() == entity_iterator->id() );
    BOOST_VERIFY( comm->getRank() == entity_iterator->ownerRank() );
    BOOST_VERIFY( space_dim == entity_iterator->topologicalDimension() );
    BOOST_VERIFY( space_dim == entity_iterator->physicalDimension() );

    BOOST_VERIFY( entity_iterator->inBlock( subdomain_1_id ) );
    BOOST_VERIFY( !entity_iterator->inBlock( subdomain_2_id ) );
    BOOST_VERIFY( entity_iterator->onBoundary( boundary_1_id ) );
    BOOST_VERIFY( !entity_iterator->onBoundary( boundary_2_id ) );

    auto extra_data_1 =
        entity_iterator->extraData();
    BOOST_VERIFY( hex_elem ==
                   Teuchos::rcp_dynamic_cast<
				   LibmeshAdapter::LibmeshEntityExtraData<libMesh::Elem>>(
                       extra_data_1 )
                       ->d_libmesh_geom.getRawPtr() );

    Teuchos::Tuple<double, 6> hex_bounds_1;
    entity_iterator->boundingBox( hex_bounds_1 );
    BOOST_VERIFY( 0.0 == hex_bounds_1[0] );
    BOOST_VERIFY( 0.0 == hex_bounds_1[1] );
    BOOST_VERIFY( 0.0 == hex_bounds_1[2] );
    BOOST_VERIFY( 1.0 == hex_bounds_1[3] );
    BOOST_VERIFY( 1.0 == hex_bounds_1[4] );
    BOOST_VERIFY( 1.0 == hex_bounds_1[5] );

    // Test the end of the iterator.
    entity_iterator++;
    BOOST_VERIFY( entity_iterator != entity_iterator.begin() );
    BOOST_VERIFY( entity_iterator == entity_iterator.end() );

    // Make an iterator with a subdomain 1 predicate.
//    DataTransferKit::BlockPredicate subdomain_1_pred(
//        Teuchos::Array<int>( 1, subdomain_1_id ) );
//    auto subdomain_1_iterator =
//    		LibmeshAdapter::LibmeshEntityIterator<
//            libMesh::Mesh::const_element_iterator>(
//            mesh->elements_begin(), mesh->elements_begin(),
//            mesh->elements_end(), mesh.ptr(),
//            Teuchos::ptrFromRef( adjacencies ),
//            subdomain_1_pred.getFunction() );
//    BOOST_VERIFY( subdomain_1_iterator.size() == num_hex );
//
//    // Make an iterator with a subdomain 2 predicate.
//    DataTransferKit::BlockPredicate subdomain_2_pred(
//        Teuchos::Array<int>( 2, subdomain_2_id ) );
//    auto subdomain_2_iterator =
//    		LibmeshAdapter::LibmeshEntityIterator<
//            libMesh::Mesh::const_element_iterator>(
//            mesh->elements_begin(), mesh->elements_begin(),
//            mesh->elements_end(), mesh.ptr(),
//            Teuchos::ptrFromRef( adjacencies ),
//            subdomain_2_pred.getFunction() );
//    BOOST_VERIFY( subdomain_2_iterator.size() == 0 );
//
//    // Make a boundary iterator for the elems.
//    DataTransferKit::BoundaryPredicate boundary_1_elem_pred(
//        Teuchos::Array<int>( 1, subdomain_1_id ) );
//    auto elem_boundary_it_1 =
//    		LibmeshAdapter::LibmeshEntityIterator<
//            libMesh::Mesh::const_element_iterator>(
//            mesh->elements_begin(), mesh->elements_begin(),
//            mesh->elements_end(), mesh.ptr(),
//            Teuchos::ptrFromRef( adjacencies ),
//            boundary_1_elem_pred.getFunction() );
//    BOOST_VERIFY( 1 == elem_boundary_it_1.size() );
//
//    // Make a boundary iterator for the elems.
//    DataTransferKit::BoundaryPredicate boundary_2_elem_pred(
//        Teuchos::Array<int>( 1, subdomain_2_id ) );
//    auto elem_boundary_it_2 =
//        LibmeshAdapter::LibmeshEntityIterator<
//            libMesh::Mesh::const_element_iterator>(
//            mesh->elements_begin(), mesh->elements_begin(),
//            mesh->elements_end(), mesh.ptr(),
//            Teuchos::ptrFromRef( adjacencies ),
//            boundary_2_elem_pred.getFunction() );
//    BOOST_VERIFY( 0 == elem_boundary_it_2.size() );

    // Make an iterator for the nodes.
    LibmeshAdapter::NodePredicateFunction all_nodes = [&](LibmeshAdapter::LibmeshEntity<libMesh::Node>) {
    	return true;
    };
    auto node_iterator =
    		LibmeshAdapter::LibmeshEntityIterator<
            libMesh::Mesh::const_node_iterator>(
            mesh->nodes_begin(), mesh->nodes_begin(), mesh->nodes_end(),
            mesh.ptr(), Teuchos::ptrFromRef( adjacencies ), all_nodes );
    BOOST_VERIFY( 8 == node_iterator.size() );

    // Make a boundary iterator for the nodes.
//    DataTransferKit::BoundaryPredicate boundary_1_node_pred(
//        Teuchos::Array<int>( 1, subdomain_1_id ) );
//    auto node_boundary_it_1 =
//    		LibmeshAdapter::LibmeshEntityIterator<
//            libMesh::Mesh::const_node_iterator>(
//            mesh->nodes_begin(), mesh->nodes_begin(), mesh->nodes_end(),
//            mesh.ptr(), Teuchos::ptrFromRef( adjacencies ),
//            boundary_1_node_pred.getFunction() );
//    BOOST_VERIFY( 0 == node_boundary_it_1.size() );
//
//    // Make a boundary iterator for the nodes.
//    DataTransferKit::BoundaryPredicate boundary_2_node_pred(
//        Teuchos::Array<int>( 1, subdomain_2_id ) );
//    auto node_boundary_it_2 =
//    		LibmeshAdapter::LibmeshEntityIterator<
//            libMesh::Mesh::const_node_iterator>(
//            mesh->nodes_begin(), mesh->nodes_begin(), mesh->nodes_end(),
//            mesh.ptr(), Teuchos::ptrFromRef( adjacencies ),
//            boundary_2_node_pred.getFunction() );
//    BOOST_VERIFY( 1 == node_boundary_it_2.size() );
}

//---------------------------------------------------------------------------//
// end tstLibmeshEntityIterator.cpp
//---------------------------------------------------------------------------//
