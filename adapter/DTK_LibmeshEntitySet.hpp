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
 * \brief LibmeshDTKAdpaters_LibmeshEntitySet.hpp
 * \author Stuart R. Slattery
 * \brief Libmesh mesh entity set.
 */
//---------------------------------------------------------------------------//

#ifndef LIBMESHDTKADAPTERS_LIBMESHENTITYSET_HPP
#define LIBMESHDTKADAPTERS_LIBMESHENTITYSET_HPP

#include <functional>
#include <unordered_map>

#include "DTK_LibmeshAdjacencies.hpp"
#include "DTK_LibmeshEntity.hpp"
#include "DTK_LibmeshEntityExtraData.hpp"
#include "DTK_LibmeshEntityIterator.hpp"

#include <DTK_Types.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OpaqueWrapper.hpp>

#include <libmesh/mesh_base.h>

namespace LibmeshAdapter
{

//---------------------------------------------------------------------------//
/*!
  \class LibmeshEntitySet
  \brief Libmesh entity set.

  Entity set implementation for Libmesh.
*/
//---------------------------------------------------------------------------//
class LibmeshEntitySet {
public:
	/*!
	 * \brief Constructor.
	 */
	LibmeshEntitySet(const Teuchos::RCP<libMesh::MeshBase> &libmesh_mesh) :
			d_libmesh_mesh(libmesh_mesh), d_adjacencies(new LibmeshAdjacencies(libmesh_mesh)) {

	}

    //@{
    //! Parallel functions.
    /*!
     * \brief Get the parallel communicator for the entity set.
     * \return A reference-counted pointer to the parallel communicator.
     */
    Teuchos::RCP<const Teuchos::Comm<int>> communicator() const {
    	   return Teuchos::rcp(
    	        new Teuchos::MpiComm<int>( d_libmesh_mesh->comm().get() ) );
    }
    //@}

    //@{
    //! Geometric data functions.
    /*!
     * \brief Return the largest physical dimension of the entities in the
     * set.
     * \return The physical dimension of the set.
     */
    int physicalDimension() const {
        return d_libmesh_mesh->mesh_dimension();
    }
    //@}

    /*!
      * \brief Get the local bounding box of entities of the set.
      *
      * \return A Cartesian box the bounds all local entities in the set.
      */
	void localBoundingBox(Teuchos::Tuple<double, 6> &bounds)  {

		double max = std::numeric_limits<double>::max();
		bounds = Teuchos::tuple(max, max, max, -max, -max, -max);
		Teuchos::Tuple<double, 6> entity_bounds;

		auto thisRank = communicator()->getRank();

		constexpr int nodeTopo = 0;
		NodePredicateFunction f =
				[&] (LibmeshEntity<libMesh::Node> ent) {
					return ent.ownerRank() == thisRank;
				};
		auto dim_it = this->entityIterator(f);
		auto entity_begin = dim_it.begin();
		auto entity_end = dim_it.end();
		for (auto entity_it = entity_begin; entity_it != entity_end;
				++entity_it) {
			entity_it->boundingBox(entity_bounds);
			for (int n = 0; n < 3; ++n) {
				bounds[n] = std::min(bounds[n], entity_bounds[n]);
				bounds[n + 3] = std::max(bounds[n + 3],
						entity_bounds[n + 3]);
			}
		}

		for (int i = 1; i < 4; ++i) {
			ElemPredicateFunction f = [&] (LibmeshEntity<libMesh::Elem> ent) {
				return ent.ownerRank() == thisRank;
			};
			auto dim_it = this->entityIterator(f);
			auto entity_begin = dim_it.begin();
			auto entity_end = dim_it.end();
			for (auto entity_it = entity_begin; entity_it != entity_end;
					++entity_it) {
				entity_it->boundingBox(entity_bounds);
				for (int n = 0; n < 3; ++n) {
					bounds[n] = std::min(bounds[n], entity_bounds[n]);
					bounds[n + 3] = std::max(bounds[n + 3],
							entity_bounds[n + 3]);
				}
			}
		}
	}

     /*!
      * \brief Get the global bounding box of entities of the set.
      *
      * Requires global communication: a single all-reduce call
      *
      * \return A Cartesian box the bounds all global entities in the set.
      */
     void globalBoundingBox( Teuchos::Tuple<double, 6> &bounds ) {
    	    double max = std::numeric_limits<double>::max();
    	    bounds = Teuchos::tuple( max, max, max, max, max, max );

    	    Teuchos::Tuple<double, 6> local_bounds;
    	    this->localBoundingBox( local_bounds );
    	    local_bounds[3] *= -1;
    	    local_bounds[4] *= -1;
    	    local_bounds[5] *= -1;

    	    Teuchos::reduceAll( *( this->communicator() ), Teuchos::REDUCE_MIN, 6,
    	                        &local_bounds[0], &bounds[0] );

    	    bounds[3] *= -1;
    	    bounds[4] *= -1;
    	    bounds[5] *= -1;

     }

    //@{
    //! Entity access functions.
    /*!
     * \brief Given an EntityId, get the entity.
     * \param entity_id Get the entity with this id.
     * \param entity The entity with the given id.
     */
//	template<typename LibmeshEntityT>
//	void getEntity(const unsigned long int entity_id,
//			const int topological_dimension, LibmeshEntityT &entity) const {
//		if (0 == topological_dimension) {
//			entity = LibmeshEntityT(
//					Teuchos::ptr(d_adjacencies->getNodeById(entity_id)),
//					d_libmesh_mesh.ptr(), d_adjacencies.ptr());
//		} else {
//			entity = LibmeshEntityT(
//					Teuchos::ptr(d_adjacencies->getElemById(entity_id)),
//					d_libmesh_mesh.ptr(), d_adjacencies.ptr());
//		}
//	}

	void getEntity(const unsigned long int entity_id, LibmeshEntity<libMesh::Node>& entity) const {
		entity = LibmeshEntity<libMesh::Node>(Teuchos::ptr(d_adjacencies->getNodeById(entity_id)),
					d_libmesh_mesh.ptr(), d_adjacencies.ptr());
	}
	void getEntity(const unsigned long int entity_id, LibmeshEntity<libMesh::Elem>& entity) const {
		entity = LibmeshEntity<libMesh::Elem>(Teuchos::ptr(d_adjacencies->getElemById(entity_id)),
							d_libmesh_mesh.ptr(), d_adjacencies.ptr());
	}

//    /*!
//     * \brief Get a iterator of the given entity type that satisfy the given
//     * predicate.
//     * \param entity_type The type of entity to get a iterator for.
//     * \param predicate The selection predicate.
//     * \return A iterator of entities of the given type.
//	 */
//	template<typename PredicateT>
//	auto entityIterator(const int topological_dimension,
//			const PredicateT &predicate) ->
//			typename std::conditional<
//				std::is_same<PredicateT, NodePredicateFunction>::value,
//				LibmeshEntityIterator<libMesh::MeshBase::const_node_iterator>,
//				LibmeshEntityIterator<libMesh::MeshBase::const_element_iterator>
//				>::type {
//
//		bool isNodePredicate = std::is_same<PredicateT, NodePredicateFunction>::value;
//
//		if (isNodePredicate) {
//	       return
//	            LibmeshEntityIterator<libMesh::MeshBase::const_node_iterator>(
//	                d_libmesh_mesh->local_nodes_begin(),
//	                d_libmesh_mesh->local_nodes_begin(),
//	                d_libmesh_mesh->local_nodes_end(), d_libmesh_mesh.ptr(),
//	                d_adjacencies.ptr(), predicate );
//	    }
//	    else
//	    {
//	        return
//	            LibmeshEntityIterator<libMesh::MeshBase::const_element_iterator>(
//	                d_libmesh_mesh->local_elements_begin(),
//	                d_libmesh_mesh->local_elements_begin(),
//	                d_libmesh_mesh->local_elements_end(), d_libmesh_mesh.ptr(),
//	                d_adjacencies.ptr(), predicate );
//	    }
//	}

	LibmeshEntityIterator<libMesh::MeshBase::const_node_iterator> entityIterator(
			const NodePredicateFunction & predicate) {
		return LibmeshEntityIterator<libMesh::MeshBase::const_node_iterator>(
				d_libmesh_mesh->local_nodes_begin(),
				d_libmesh_mesh->local_nodes_begin(),
				d_libmesh_mesh->local_nodes_end(), d_libmesh_mesh.ptr(),
				d_adjacencies.ptr(), predicate);
	}

	LibmeshEntityIterator<libMesh::MeshBase::const_element_iterator> entityIterator(
			const ElemPredicateFunction & predicate) {
		return LibmeshEntityIterator<libMesh::MeshBase::const_element_iterator>(
				d_libmesh_mesh->local_elements_begin(),
				d_libmesh_mesh->local_elements_begin(),
				d_libmesh_mesh->local_elements_end(), d_libmesh_mesh.ptr(),
				d_adjacencies.ptr(), predicate);
	}
	/*!
	 * \brief Given an entity, get the entities of the given type that are
     * adjacent to it.
     */

	void getAdjacentEntities(const LibmeshEntity<libMesh::Node>& entity,

			Teuchos::Array<LibmeshEntity<libMesh::Elem>> & adjacent_entities) const {
		getAdjacentEntitiesImpl<libMesh::Node, libMesh::Elem>(entity, adjacent_entities);
	}

	void getAdjacentEntities(const LibmeshEntity<libMesh::Elem>& entity,
			Teuchos::Array<LibmeshEntity<libMesh::Node>> & adjacent_entities) const {
		getAdjacentEntitiesImpl<libMesh::Elem, libMesh::Node>(entity, adjacent_entities);
	}

	void getAdjacentEntities(const LibmeshEntity<libMesh::Elem>& entity,
			Teuchos::Array<LibmeshEntity<libMesh::Elem>> & adjacent_entities) const {
		getAdjacentEntitiesImpl<libMesh::Elem, libMesh::Elem>(entity, adjacent_entities);
	}

	void getAdjacentEntities(const LibmeshEntity<libMesh::Node>& entity,
			Teuchos::Array<LibmeshEntity<libMesh::Node>> & adjacent_entities) const {
		adjacent_entities.clear();
	}

  private:
    template <class FromGeomType, class ToGeomType>
    void getAdjacentEntitiesImpl(
        const LibmeshEntity<FromGeomType> &entity,
        Teuchos::Array<LibmeshEntity<ToGeomType>> &adjacent_entities ) const;

  private:
    // Libmesh mesh.
    Teuchos::RCP<libMesh::MeshBase> d_libmesh_mesh;

    // Mesh adjacencies.
    Teuchos::RCP<LibmeshAdjacencies> d_adjacencies;
};

//---------------------------------------------------------------------------//
// Template functions.
//---------------------------------------------------------------------------//

template <class FromGeomType, class ToGeomType>
void LibmeshEntitySet::getAdjacentEntitiesImpl(
    const LibmeshEntity<FromGeomType> &entity,
    Teuchos::Array<LibmeshEntity<ToGeomType>> &adjacent_entities ) const
{
    Teuchos::Array<Teuchos::Ptr<ToGeomType>> adjacent_libmesh;
    d_adjacencies->getLibmeshAdjacencies(
        Teuchos::rcp_dynamic_cast<LibmeshEntityExtraData<FromGeomType>>(
            entity.extraData() )
            ->d_libmesh_geom,
        adjacent_libmesh );

    adjacent_entities.resize( adjacent_libmesh.size() );
    typename Teuchos::Array<Teuchos::Ptr<ToGeomType>>::iterator libmesh_it;
    typename Teuchos::Array<LibmeshEntity<ToGeomType>>::iterator dtk_it;
    for ( libmesh_it = adjacent_libmesh.begin(),
          dtk_it = adjacent_entities.begin();
          libmesh_it != adjacent_libmesh.end(); ++libmesh_it, ++dtk_it )
    {
        *dtk_it = LibmeshEntity<ToGeomType>( *libmesh_it, d_libmesh_mesh.ptr(),
                                             d_adjacencies.ptr() );
    }
}


// Get adjacent entities implementation.
//template <typename LibmeshEntityT, typename LibmeshEntityS, class FromGeomType, class ToGeomType>
//void LibmeshEntitySet::getAdjacentEntitiesImpl(
//    const LibmeshEntityT &entity,
//    Teuchos::Array<LibmeshEntityS> &adjacent_entities ) const
//{
//    Teuchos::Array<Teuchos::Ptr<ToGeomType>> adjacent_libmesh;
//
//    auto extraData = entity.extraData();
//
//	d_adjacencies->getLibmeshAdjacencies(extraData->d_libmesh_geom,
//			adjacent_libmesh);
//
////    d_adjacencies->getLibmeshAdjacencies(
////        Teuchos::rcp_dynamic_cast<LibmeshEntityExtraData<FromGeomType>>(
////            entity.extraData() )
////            ->d_libmesh_geom,
////        adjacent_libmesh );
//
//    adjacent_entities.resize( adjacent_libmesh.size() );
//    auto libmesh_it = adjacent_libmesh.begin();
//    auto dtk_it = adjacent_entities.begin();
////    typename Teuchos::Array<LibmeshEntity<ToGeomType>>::iterator dtk_it;
//    for ( libmesh_it = adjacent_libmesh.begin(),
//          dtk_it = adjacent_entities.begin();
//          libmesh_it != adjacent_libmesh.end(); ++libmesh_it, ++dtk_it )
//    {
//        *dtk_it = LibmeshEntityS( *libmesh_it, d_libmesh_mesh.ptr(),
//                                             d_adjacencies.ptr() );
//    }
//}


//---------------------------------------------------------------------------//

} // end namespace LibmeshAdapter

#endif // end LIBMESHDTKADAPTERS_LIBMESHENTITYSET_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshEntitySet.hpp
//---------------------------------------------------------------------------//
