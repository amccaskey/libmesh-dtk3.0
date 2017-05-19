#ifndef LIBMESHDTKADAPTERS_ADJACENCIES_HPP
#define LIBMESHDTKADAPTERS_ADJACENCIES_HPP

#include <unordered_map>

#include <Types.hpp>
#include <DTK_DBC.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_RCP.hpp>

#include <libmesh/elem.h>
#include <libmesh/mesh_base.h>
#include <libmesh/node.h>

namespace LibmeshAdapter {
//---------------------------------------------------------------------------//
/*!
 \class LibmeshAdjacencies
 \brief Libmesh adjacency information.

 Libmesh doesn't create the upward adjacency graph so this class takes care
 of that. For now, only element and node adjacencies are supported in this
 implementation. This will be sufficient for shared-domain coupling. For
 surface transfers, we will need to add support for edges and faces.
 */
//---------------------------------------------------------------------------//
class LibmeshAdjacencies {
public:
	// Constructor.
	LibmeshAdjacencies(const Teuchos::RCP<libMesh::MeshBase> &mesh) :
			d_mesh(mesh) {
		// Map nodes to elements and elements to their ids.
		int num_nodes = 0;
		libMesh::MeshBase::element_iterator elem_begin =
				d_mesh->local_elements_begin();
		libMesh::MeshBase::element_iterator elem_end =
				d_mesh->local_elements_end();
		for (auto elem = elem_begin; elem != elem_end; ++elem) {
			d_elem_id_map.emplace((*elem)->id(), *elem);
			num_nodes = (*elem)->n_nodes();
			for (int n = 0; n < num_nodes; ++n) {
				d_node_to_elem_map.emplace((*elem)->get_node(n), *elem);
			}
		}

		// Map nodes to their ids.
		libMesh::MeshBase::node_iterator node_begin =
				d_mesh->local_nodes_begin();
		libMesh::MeshBase::node_iterator node_end = d_mesh->local_nodes_end();
		for (auto node = node_begin; node != node_end; ++node) {
			d_node_id_map.emplace((*node)->id(), *node);
		}
	}

	// Get the adjacency of a libmesh geom object.
	template<class FromGeomType, class ToGeomType>
	void getLibmeshAdjacencies(const Teuchos::Ptr<FromGeomType> &entity,
			Teuchos::Array<Teuchos::Ptr<ToGeomType>> &adjacent_entities) const;

	// Given a node global id get its pointer.
	libMesh::Node *getNodeById(const unsigned long int id) const {
		DTK_REQUIRE(d_node_id_map.count(id));
		return d_node_id_map.find(id)->second;
	}

	// Given a elem global id get its pointer.
	libMesh::Elem *getElemById(const unsigned long int id) const {
		DTK_REQUIRE(d_elem_id_map.count(id));
		return d_elem_id_map.find(id)->second;
	}

private:
	// libMesh mesh.
	Teuchos::RCP<libMesh::MeshBase> d_mesh;

	// Node-to-element map.
	std::unordered_multimap<libMesh::Node *, libMesh::Elem *> d_node_to_elem_map;

	// Id-to-node map.
	std::unordered_map<unsigned long int, libMesh::Node *> d_node_id_map;

	// Id-to-elem map.
	std::unordered_map<unsigned long int, libMesh::Elem *> d_elem_id_map;
};

//---------------------------------------------------------------------------//

template<>
void LibmeshAdjacencies::getLibmeshAdjacencies<libMesh::Node, libMesh::Elem>(
		const Teuchos::Ptr<libMesh::Node> &entity,
		Teuchos::Array<Teuchos::Ptr<libMesh::Elem>> &adjacent_entities) const {
	auto elem_range = d_node_to_elem_map.equal_range(entity.getRawPtr());
	int num_elem = std::distance(elem_range.first, elem_range.second);
	adjacent_entities.resize(num_elem);
	int e = 0;
	for (auto node_elems = elem_range.first; node_elems != elem_range.second;
			++node_elems, ++e) {
		adjacent_entities[e] = Teuchos::ptr(node_elems->second);
	}
}

//---------------------------------------------------------------------------//
/*!
 * Get the adjacency of a libmesh geom object. Elem to node overload.
 */
template<>
void LibmeshAdjacencies::getLibmeshAdjacencies<libMesh::Elem, libMesh::Node>(
		const Teuchos::Ptr<libMesh::Elem> &entity,
		Teuchos::Array<Teuchos::Ptr<libMesh::Node>> &adjacent_entities) const {
	int num_nodes = entity->n_nodes();
	adjacent_entities.resize(num_nodes);
	for (int n = 0; n < num_nodes; ++n) {
		adjacent_entities[n] = Teuchos::ptr(entity->get_node(n));
	}
}

//---------------------------------------------------------------------------//
/*!
 * Get the adjacency of a libmesh geom object. Node to node overload.
 */
template<>
void LibmeshAdjacencies::getLibmeshAdjacencies<libMesh::Node, libMesh::Node>(
		const Teuchos::Ptr<libMesh::Node> & /*entity*/,
		Teuchos::Array<Teuchos::Ptr<libMesh::Node>> &adjacent_entities) const {
	adjacent_entities.clear();
}

//---------------------------------------------------------------------------//
/*!
 * Get the adjacency of a libmesh geom object. Elem to elem overload.
 */
template<>
void LibmeshAdjacencies::getLibmeshAdjacencies<libMesh::Elem, libMesh::Elem>(
		const Teuchos::Ptr<libMesh::Elem> & /*entity*/,
		Teuchos::Array<Teuchos::Ptr<libMesh::Elem>> &adjacent_entities) const {
	adjacent_entities.clear();
}

} // end namespace LibmeshAdapter

#endif // end LIBMESHDTKADAPTERS_ADJACENCIES_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshAdjacencies.hpp
//---------------------------------------------------------------------------//
