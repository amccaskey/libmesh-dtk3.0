#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LibmeshNodalShapeFunctionTester
#include <boost/test/included/unit_test.hpp>

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "LibmeshAdjacencies.hpp"
#include "LibmeshEntity.hpp"
#include "LibmeshNodalShapeFunction.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_Ptr.hpp"
#include "Teuchos_RCP.hpp"
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
BOOST_AUTO_TEST_CASE( checkLibmeshNodalShapeFunction )
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
    BOOST_ASSERT( libMesh::initialized() );
    BOOST_ASSERT( (int)libmesh_init.comm().rank() == comm->getRank() );
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
    for ( int i = 0; i < 8; ++i )
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

    // Make a libmesh system. We will put a first order linear basis on the
    // elements.
    libMesh::EquationSystems equation_systems( *mesh );
    libMesh::LinearImplicitSystem &system =
        equation_systems.add_system<libMesh::LinearImplicitSystem>( "Test" );
    system.add_variable( "test_var", libMesh::FIRST );

    // Create a shape function.
    auto shape_function =
        Teuchos::rcp( new LibmeshAdapter::LibmeshNodalShapeFunction(
            mesh, Teuchos::rcpFromRef( system ) ) );

    // Test the shape function dof ids for the hex.
    Teuchos::Array<unsigned long int> dof_ids;
    shape_function->entitySupportIds( dtk_entity, dof_ids );
    BOOST_ASSERT( num_nodes == dof_ids.size() );
    for ( int n = 0; n < num_nodes; ++n )
    {
    	BOOST_ASSERT( dof_ids[n] == nodes[n]->id() );
    }

    // Test the value evaluation for the hex.
    Teuchos::Array<double> ref_point( space_dim, 0.0 );
    Teuchos::Array<double> values;
    shape_function->evaluateValue( dtk_entity, ref_point(), values );
    BOOST_ASSERT( values.size() == num_nodes );
    for ( int n = 0; n < num_nodes; ++n )
    {
    	BOOST_ASSERT( values[n] == 1.0 / num_nodes );
    }
    ref_point[0] = -1.0;
    ref_point[1] = -1.0;
    ref_point[2] = -1.0;
    shape_function->evaluateValue( dtk_entity, ref_point(), values );
    BOOST_ASSERT( values.size() == num_nodes );
    BOOST_ASSERT( values[0] == 1.0 );
    for ( int n = 1; n < num_nodes; ++n )
    {
    	BOOST_ASSERT( values[n] == 0.0 );
    }

    // Test the shape function dof ids for the nodes.
    for ( int n = 0; n < num_nodes; ++n )
    {
        dof_ids.clear();
       auto dtk_node =
            LibmeshAdapter::LibmeshEntity<libMesh::Node>(
                Teuchos::ptr( nodes[n] ), mesh.ptr(),
                Teuchos::ptrFromRef( adjacencies ) );
        shape_function->entitySupportIds( dtk_node, dof_ids );
        BOOST_ASSERT( dof_ids.size() == 1 );
        BOOST_ASSERT( dof_ids[0] == nodes[n]->id() );
    }
}

//---------------------------------------------------------------------------//
// end of tstLibmeshNodalShapeFunction.cpp
//---------------------------------------------------------------------------//
