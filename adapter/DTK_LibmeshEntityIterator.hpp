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
 * \brief DTK_LibmeshEntityIterator.hpp
 * \author Stuart R. Slattery
 * \brief Entity iterator interface.
 */
//---------------------------------------------------------------------------//
#ifndef LIBMESHDTKADAPTERS_LIBMESHENTITYITERATOR_HPP
#define LIBMESHDTKADAPTERS_LIBMESHENTITYITERATOR_HPP

#include <functional>
#include <iterator>
#include <vector>

#include "DTK_LibmeshAdjacencies.hpp"

#include <DTK_Types.hpp>

#include <Teuchos_Ptr.hpp>

#include <libmesh/elem.h>
#include <libmesh/mesh_base.h>

namespace LibmeshAdapter {

using NodePredicateFunction = std::function<bool(LibmeshEntity<libMesh::Node>)>;
using ElemPredicateFunction = std::function<bool(LibmeshEntity<libMesh::Elem>)>;


template<class LibmeshGeomIterator>
class LibmeshEntityIterator: public std::iterator<std::forward_iterator_tag,
		LibmeshEntity<
				typename std::remove_pointer<
						typename LibmeshGeomIterator::value_type>::type>> {
public:

	using BaseIterType = LibmeshEntityIterator<LibmeshGeomIterator>;
	using LibmeshEntityType = LibmeshEntity<typename std::remove_pointer<typename LibmeshGeomIterator::value_type>::type>;
	using PredicateFunction = typename std::conditional<std::is_same<LibmeshEntityType, LibmeshEntity<libMesh::Node>>::value,
			NodePredicateFunction, ElemPredicateFunction>::type;

	/*!
    * \brief Constructor.
    */
	LibmeshEntityIterator() {}

   /*!
    * \brief Copy constructor.
    */
	LibmeshEntityIterator( const BaseIterType &rhs ) :
		d_libmesh_iterator( rhs.d_libmesh_iterator ),
			d_libmesh_iterator_begin( rhs.d_libmesh_iterator_begin ),
			d_libmesh_iterator_end( rhs.d_libmesh_iterator_end ),
			d_libmesh_mesh( rhs.d_libmesh_mesh ),
			d_adjacencies( rhs.d_adjacencies ) {
		this->b_predicate = rhs.b_predicate;
	}

	LibmeshEntityIterator( LibmeshGeomIterator libmesh_iterator,
	                       LibmeshGeomIterator libmesh_iterator_begin,
	                       LibmeshGeomIterator libmesh_iterator_end,
	                       const Teuchos::Ptr<libMesh::MeshBase> &libmesh_mesh,
	                       const Teuchos::Ptr<LibmeshAdjacencies> &adjacencies,
	                       const PredicateFunction &predicate ) : d_libmesh_iterator( libmesh_iterator ),
	                    		   d_libmesh_iterator_begin( libmesh_iterator_begin ),
								   d_libmesh_iterator_end( libmesh_iterator_end ),
								   d_libmesh_mesh( libmesh_mesh ),
								   d_adjacencies( adjacencies ) {
		this->b_predicate = predicate;
	}

   /*!
    * \brief Assignment operator.
    */
	BaseIterType &operator=( const BaseIterType &rhs ) {
	    this->b_predicate = rhs.b_predicate;
	    if ( &rhs == this )
	    {
	        return *this;
	    }
	    d_libmesh_iterator = rhs.d_libmesh_iterator;
	    d_libmesh_iterator_begin = rhs.d_libmesh_iterator_begin;
	    d_libmesh_iterator_end = rhs.d_libmesh_iterator_end;
	    d_libmesh_mesh = rhs.d_libmesh_mesh;
	    d_adjacencies = rhs.d_adjacencies;
	    return *this;
	}

   /*!
    * \brief Destructor.
    */
   virtual ~LibmeshEntityIterator() {}

   // Pre-increment operator.
   virtual BaseIterType &operator++() {
	   ++d_libmesh_iterator;
	   return *this;
   }

   // Post-increment operator.
   virtual BaseIterType operator++( int ) {
	   DTK_REQUIRE( b_iterator_impl );
	   DTK_REQUIRE( *b_iterator_impl != b_iterator_impl->end() );

	   const BaseIterType tmp( *this );
	   increment();
	   return tmp;
   }

   // Dereference operator.
   virtual LibmeshEntityType &operator*( void ) {
	   this->operator->();
	   return d_current_entity;
   }

   // Dereference operator.
   virtual LibmeshEntityType *operator->( void ) {
	    d_current_entity = LibmeshEntityType(
	        Teuchos::ptr( *d_libmesh_iterator ), d_libmesh_mesh, d_adjacencies );
	    return &d_current_entity;
   }

   // Equal comparison operator.
   virtual bool operator==( const BaseIterType &rhs ) const {
	    const BaseIterType *rhs_it =
	        static_cast<const BaseIterType *>( &rhs );
	    const BaseIterType *rhs_it_impl =
	        static_cast<const BaseIterType *>(
	            rhs_it->b_iterator_impl.get() );
	    return ( rhs_it_impl->d_libmesh_iterator == d_libmesh_iterator );
   }

   // Not equal comparison operator.
   virtual bool operator!=( const BaseIterType &rhs ) const {
	    const BaseIterType *rhs_it =
	        static_cast<const BaseIterType *>( &rhs );
	    const BaseIterType *rhs_it_impl =
	        static_cast<const BaseIterType *>(
	            rhs_it->b_iterator_impl.get() );
	    return ( rhs_it_impl->d_libmesh_iterator != d_libmesh_iterator );
   }

   // Number of elements in the iterator that meet the predicate criteria.
   std::size_t size() const {
	   std::size_t size = 0;
	    if ( b_iterator_impl )
	    {
	        size = std::distance( this->begin(), this->end() );
	    }
	    return size;
   }

   // An iterator assigned to the first valid element in the iterator.
   virtual BaseIterType begin() const {
	    return BaseIterType( d_libmesh_iterator_begin,
	                                  d_libmesh_iterator_begin,
	                                  d_libmesh_iterator_end, d_libmesh_mesh,
	                                  d_adjacencies, this->b_predicate );
   }

   // An iterator assigned to the end of all elements under the iterator.
   virtual BaseIterType end() const {
	    return BaseIterType( d_libmesh_iterator_end,
	                                  d_libmesh_iterator_begin,
	                                  d_libmesh_iterator_end, d_libmesh_mesh,
	                                  d_adjacencies, this->b_predicate );
   }

 protected:
   // Implementation.
   std::unique_ptr<BaseIterType> b_iterator_impl;

   // Predicate.
   PredicateFunction b_predicate;

 protected:
   // Create a clone of the iterator. We need this for the copy constructor
   // and assignment operator to pass along the underlying implementation
   // without pointing to the same implementation in every instance of the
   // iterator.
   virtual std::unique_ptr<BaseIterType> clone() const {
	    return std::unique_ptr<BaseIterType>(
	        new BaseIterType( *this ) );
   }

 private:

   // Libmesh iterator.
      LibmeshGeomIterator d_libmesh_iterator;

      // Libmesh iterator to the beginning of the range.
      LibmeshGeomIterator d_libmesh_iterator_begin;

      // Libmesh iterator to the end of the range.
      LibmeshGeomIterator d_libmesh_iterator_end;

      // The mesh owning the entities.
      Teuchos::Ptr<libMesh::MeshBase> d_libmesh_mesh;

      // Mesh adjacencies.
      Teuchos::Ptr<LibmeshAdjacencies> d_adjacencies;

      // Current entity.
      LibmeshEntityType d_current_entity;

   // Advance the iterator to the first valid element that satisfies the
   // predicate or the end of the iterator.
   void advanceToFirstValidElement() {
	    DTK_REQUIRE( b_iterator_impl );
	    if ( ( *this != end() ) && !b_predicate( **this ) )
	    {
	        increment();
	    }
   }

   // Increment the iterator implementation forward until either a valid
   // increment is found or we have reached the end.
   void increment() {
	    DTK_REQUIRE( b_iterator_impl );
	    DTK_REQUIRE( *b_iterator_impl != b_iterator_impl->end() );

	    // Apply the increment operator.
	    BaseIterType &it = b_iterator_impl->operator++();

	    // Get the end of the range.
	    BaseIterType end = b_iterator_impl->end();

	    // If the we are not at the end or the predicate is not satisfied by the
	    // current element, increment until either of these conditions is
	    // satisfied.
	    while ( it != end && !b_predicate( *it ) )
	    {
	        it = b_iterator_impl->operator++();
	    }

   }
};

//---------------------------------------------------------------------------//

}
	// end namespace LibmeshAdapter

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

//#include "DTK_LibmeshEntityIterator_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end LIBMESHDTKADAPTERS_LIBMESHENTITYITERATOR_HPP

//---------------------------------------------------------------------------//
// end DTK_LibmeshEntityIterator.hpp
//---------------------------------------------------------------------------//
