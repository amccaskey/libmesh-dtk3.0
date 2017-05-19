//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \file tstLibmeshEntitySet.cpp
 * \author Stuart R. Slattery
 * \brief LibmeshEntitySet unit tests.
 */
//---------------------------------------------------------------------------//

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshEntitySetTester
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include <DTK_LibmeshEntityExtraData.hpp>
#include <DTK_LibmeshEntitySet.hpp>

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
BOOST_AUTO_TEST_CASE( checkLibmeshEntitySet )
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


    // Create an entity set.
    auto entity_set =
        Teuchos::rcp( new LibmeshAdapter::LibmeshEntitySet( mesh ) );

    // Test the set.
    Teuchos::RCP<const Teuchos::Comm<int>> set_comm =
        entity_set->communicator();
    BOOST_VERIFY( set_comm->getRank() == comm->getRank() );
    BOOST_VERIFY( set_comm->getSize() == comm->getSize() );
    BOOST_VERIFY( space_dim == entity_set->physicalDimension() );

    // Make an iterator for the hex.
    LibmeshAdapter::ElemPredicateFunction all_pred =
        [=]( LibmeshAdapter::LibmeshEntity<libMesh::Elem> ) { return true; };
    auto volume_iterator =
        entity_set->entityIterator( all_pred );

    // Test the volume iterator.
    BOOST_VERIFY( volume_iterator.size() == 1 );
    BOOST_VERIFY( volume_iterator == volume_iterator.begin() );
    BOOST_VERIFY( volume_iterator != volume_iterator.end() );

    // Test the volume under the iterator.
    BOOST_VERIFY( hex_elem->id() == volume_iterator->id() );
    BOOST_VERIFY( comm->getRank() == volume_iterator->ownerRank() );
    BOOST_VERIFY( space_dim == volume_iterator->topologicalDimension() );
    BOOST_VERIFY( space_dim == volume_iterator->physicalDimension() );

   auto extra_data_1 =
        volume_iterator->extraData();
   BOOST_VERIFY( hex_elem ==
                   Teuchos::rcp_dynamic_cast<
                       LibmeshAdapter::LibmeshEntityExtraData<libMesh::Elem>>(
                       extra_data_1 )
                       ->d_libmesh_geom.getRawPtr() );

    Teuchos::Tuple<double, 6> hex_bounds_1;
    volume_iterator->boundingBox( hex_bounds_1 );
    BOOST_VERIFY( 0.0 == hex_bounds_1[0] );
    BOOST_VERIFY( 0.0 == hex_bounds_1[1] );
    BOOST_VERIFY( 0.0 == hex_bounds_1[2] );
    BOOST_VERIFY( 1.0 == hex_bounds_1[3] );
    BOOST_VERIFY( 1.0 == hex_bounds_1[4] );
    BOOST_VERIFY( 1.0 == hex_bounds_1[5] );

    // Test the end of the iterator.
    volume_iterator++;
    BOOST_VERIFY( volume_iterator != volume_iterator.begin() );
    BOOST_VERIFY( volume_iterator == volume_iterator.end() );

    // Make an iterator for the nodes.
    LibmeshAdapter::NodePredicateFunction all_node_pred =
            [=]( LibmeshAdapter::LibmeshEntity<libMesh::Node> ) { return true; };
   auto node_iterator =
        entity_set->entityIterator( all_node_pred );

    // Test the node iterator.
    unsigned num_nodes = 8;
    BOOST_VERIFY( node_iterator.size() == num_nodes );
    BOOST_VERIFY( node_iterator == node_iterator.begin() );
    BOOST_VERIFY( node_iterator != node_iterator.end() );
    auto node_begin = node_iterator.begin();
    auto node_end = node_iterator.end();
    auto node_id_it = nodes.begin();
    for ( node_iterator = node_begin; node_iterator != node_end;
          ++node_iterator, ++node_id_it )
    {
    	BOOST_VERIFY( node_iterator->id() == ( *node_id_it )->id() );
    }

    // Get each entity and check.
    LibmeshAdapter::LibmeshEntity<libMesh::Elem> set_hex;
    entity_set->getEntity( hex_elem->id(), set_hex );
    BOOST_VERIFY( set_hex.id() == hex_elem->id() );

    for ( unsigned i = 0; i < num_nodes; ++i )
    {
    	LibmeshAdapter::LibmeshEntity<libMesh::Node> set_node;
        entity_set->getEntity( nodes[i]->id(), set_node );
        BOOST_VERIFY( set_node.id() == nodes[i]->id() );
    }

    // Check the adjacency function.
    Teuchos::Array<LibmeshAdapter::LibmeshEntity<libMesh::Elem>> hex_adjacent_volumes;
    entity_set->getAdjacentEntities( set_hex, hex_adjacent_volumes );
    BOOST_VERIFY( 0 == hex_adjacent_volumes.size() );

    Teuchos::Array<LibmeshAdapter::LibmeshEntity<libMesh::Node>> hex_adjacent_nodes;
    entity_set->getAdjacentEntities( set_hex, hex_adjacent_nodes );
    BOOST_VERIFY( num_nodes == hex_adjacent_nodes.size() );
    for ( unsigned i = 0; i < num_nodes; ++i )
    {
    	BOOST_VERIFY( hex_adjacent_nodes[i].id() == nodes[i]->id() );
    }

    for ( unsigned i = 0; i < num_nodes; ++i )
    {
        Teuchos::Array<LibmeshAdapter::LibmeshEntity<libMesh::Elem>> node_adjacent_volumes;
        entity_set->getAdjacentEntities( hex_adjacent_nodes[i],
                                         node_adjacent_volumes );
        BOOST_VERIFY( 1 == node_adjacent_volumes.size() );
        BOOST_VERIFY( node_adjacent_volumes[0].id() == hex_elem->id() );
    }

}

//---------------------------------------------------------------------------//
// end tstLibmeshEntitySet.cpp
//---------------------------------------------------------------------------//
