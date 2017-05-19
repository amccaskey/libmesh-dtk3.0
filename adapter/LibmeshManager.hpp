#ifndef LIBMESHDTKADAPTERS_LIBMESHMANAGER_HPP
#define LIBMESHDTKADAPTERS_LIBMESHMANAGER_HPP

#include <functional>
#include <string>

#include "LibmeshFieldMultiVector.hpp"
#include "LibmeshEntityLocalMap.hpp"
#include "LibmeshNodalShapeFunction.hpp"
#include "LibmeshEntityIntegrationRule.hpp"
#include "LibmeshEntityIterator.hpp"

#include "Types.hpp"

#include <Teuchos_RCP.hpp>

#include <libmesh/mesh_base.h>
#include <libmesh/system.h>

namespace LibmeshAdapter {

class FunctionSpace {
public:
	/*!
	 * \brief Constructor.
	 */
	FunctionSpace(const Teuchos::RCP<LibmeshEntitySet> &entity_set,
			const Teuchos::RCP<LibmeshEntityLocalMap> &local_map,
			const Teuchos::RCP<LibmeshNodalShapeFunction> &shape_function,
			const Teuchos::RCP<LibmeshEntityIntegrationRule> &integration_rule) :
			d_entity_set(entity_set), d_local_map(local_map), d_shape_function(
					shape_function), d_integration_rule(integration_rule) {

	}
//                   const PredicateFunction &select_function = selectAll );

	/*!
	 * \brief Get the entity set over which the fields are defined.
	 */
	Teuchos::RCP<LibmeshEntitySet> entitySet() const {
		return d_entity_set;
	}

	/*!
	 * \brief Get the local map for entities supporting the function.
	 */
	Teuchos::RCP<LibmeshEntityLocalMap> localMap() const {
		return d_local_map;
	}

	/*!
	 * \brief Get the shape function for entities supporting the function.
	 */
	Teuchos::RCP<LibmeshNodalShapeFunction> shapeFunction() const {
		return d_shape_function;
	}

	/*!
	 * \brief Get the integration rule for entities supporting the function.
	 */
	Teuchos::RCP<LibmeshEntityIntegrationRule> integrationRule() const {
		return d_integration_rule;
	}

	/*!
	 * \brief Get the selector function.
	 */
//    PredicateFunction selectFunction() const;
	/*!
	 * \brief Default select function.
	 */
//    static inline bool selectAll( Entity ) { return true; }
private:
	// The entity set over which the function space is constructed.
	Teuchos::RCP<LibmeshEntitySet> d_entity_set;

	// The reference frame for entities in the set.
	Teuchos::RCP<LibmeshEntityLocalMap> d_local_map;

	// The shape function for the entities in the set.
	Teuchos::RCP<LibmeshNodalShapeFunction> d_shape_function;

	// The integration rule for the entities in the set.
	Teuchos::RCP<LibmeshEntityIntegrationRule> d_integration_rule;

	// The selector function.
//    PredicateFunction d_select_function;
};

//---------------------------------------------------------------------------//
/*!
 \class LibmeshManager
 \brief High-level manager for Libmesh mesh.

 This manager provides a high-level class for automated construction of 
 interface objects. A user is not required to use this class but rather could
 use it to reduce code for certain implementations.
 */
//---------------------------------------------------------------------------//
class LibmeshManager {
public:
	/*!
	 * \brief Default constructor.
	 *
	 * \param libmesh_mesh Libmesh mesh.
	 *
	 * \param libsystem_system Libsystem system.
	 *
	 * \param entity_type The type of entities in the mesh that will be
	 * mapped.
	 */
	LibmeshManager(const Teuchos::RCP<libMesh::MeshBase> &libmesh_mesh,
			const Teuchos::RCP<libMesh::System> &libmesh_system) :
			d_mesh(libmesh_mesh), d_system(libmesh_system) {
		//    SelectAllPredicate pred;
		buildFunctionSpace(); //pred.getFunction() );
	}

	/*!
	 * \brief Subdomain constructor.
	 *
	 * \param libmesh_mesh Libmesh mesh.
	 *
	 * \param libsystem_system Libsystem system.
	 *
	 * \param subdomain_ids The subdomain ids to map.
	 *
	 * \param entity_type The type of entities in the mesh that will be
	 * mapped.
	 */
	LibmeshManager(const Teuchos::RCP<libMesh::MeshBase> &libmesh_mesh,
			const Teuchos::RCP<libMesh::System> &libmesh_system,
			const Teuchos::Array<libMesh::subdomain_id_type> &subdomain_ids) {}

	/*!
	 * \brief Boundary constructor.
	 *
	 * \param libmesh_mesh Libmesh mesh.
	 *
	 * \param libsystem_system Libsystem system.
	 *
	 * \param boundary_ids The boundary ids to map.
	 *
	 * \param entity_type The type of entities in the mesh that will be
	 * mapped.
	 */
	LibmeshManager(const Teuchos::RCP<libMesh::MeshBase> &libmesh_mesh,
			const Teuchos::RCP<libMesh::System> &libmesh_system,
			const Teuchos::Array<libMesh::boundary_id_type> &boundary_ids) {}

	/*!
	 * \brief Get the function space over which the mesh and its fields are
	 * defined.
	 */
	Teuchos::RCP<FunctionSpace> functionSpace() const {
		return d_function_space;
	}

	/*!
	 * \brief Given a variable name, build a field vector.
	 */
	Teuchos::RCP<LibmeshFieldMultiVector> createFieldMultiVector(
			const std::string &variable_name) {
		auto field = Teuchos::rcp(
				new LibmeshVariableField(d_mesh, d_system, variable_name));
		return Teuchos::rcp(
				new LibmeshFieldMultiVector(field,
						d_function_space->entitySet()));
	}
//@{
//! ClientManager interface implementation.
	/*!
	 * \brief Get the entity set over which the fields are defined.
	 */
	Teuchos::RCP<LibmeshEntitySet> entitySet() const {
		return d_function_space->entitySet();
	}

	/*!
	 * \brief Get the local map for entities supporting the function.
	 */
	Teuchos::RCP<LibmeshEntityLocalMap> localMap() const{
		return d_function_space->localMap();
	}

	/*!
	 * \brief Get the shape function for entities supporting the function.
	 */
	Teuchos::RCP<LibmeshNodalShapeFunction> shapeFunction() const {
		return d_function_space->shapeFunction();
	}

	/*!
	 * \brief Get the integration rule for entities supporting the function.
	 */
	Teuchos::RCP<LibmeshEntityIntegrationRule> integrationRule() const {
		return d_function_space->integrationRule();
	}

	/*!
	 * \brief Get the selector function.
	 */
//	template<typename LibmeshGeom>
//	std::function<bool(LibmeshEntity<LibmeshGeom>)> selectFunction() const;

	/*!
	 * \brief Get the field for the given string key.
	 */
	Teuchos::RCP<LibmeshVariableField> field(
			const std::string &field_name) const {
		 return Teuchos::rcp(
		        new LibmeshVariableField( d_mesh, d_system, field_name ) );
	}
	//@}

private:
	// Build the function space.
//    template<typename LibmeshGeom>
	void buildFunctionSpace() { //const std::function<bool(LibmeshEntity<LibmeshGeom>)> &pred ); {
		Teuchos::RCP<LibmeshEntitySet> entity_set = Teuchos::rcp(
				new LibmeshEntitySet(d_mesh));

		Teuchos::RCP<LibmeshEntityLocalMap> local_map = Teuchos::rcp(
				new LibmeshEntityLocalMap(d_mesh, d_system));

		Teuchos::RCP<LibmeshNodalShapeFunction> shape_function = Teuchos::rcp(
				new LibmeshNodalShapeFunction(d_mesh, d_system));

		Teuchos::RCP<LibmeshEntityIntegrationRule> integration_rule =
				Teuchos::rcp(new LibmeshEntityIntegrationRule());

		d_function_space = Teuchos::rcp(
				new FunctionSpace(entity_set, local_map, shape_function,
						integration_rule));
	}

private:
	// The mesh.
	Teuchos::RCP<libMesh::MeshBase> d_mesh;

	// The system.
	Teuchos::RCP<libMesh::System> d_system;

	// The function space over which the mesh and its fields are defined.
	Teuchos::RCP<FunctionSpace> d_function_space;
};

//---------------------------------------------------------------------------//

}// end namespace LibmeshAdapter

//---------------------------------------------------------------------------//

#endif // end LIBMESHDTKADAPTERS_LIBMESHMANAGER_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshManager.hpp
//---------------------------------------------------------------------------//
