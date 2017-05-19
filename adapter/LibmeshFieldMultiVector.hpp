#ifndef DTK_LIBMESHFIELDMULTIVECTOR_HPP
#define DTK_LIBMESHFIELDMULTIVECTOR_HPP

#include "Types.hpp"
#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#include "LibmeshVariableField.hpp"
#include "LibmeshEntitySet.hpp"

#include <Tpetra_MultiVector.hpp>

namespace LibmeshAdapter {
//---------------------------------------------------------------------------//
/*!
 \class FieldMultiVector
 \brief MultiVector interface.

 FieldMultiVector provides a Tpetra::MultiVector wrapper around application
 field data. Client implementations of the Field interface provide read/write
 access to field data on an entity-by-entity basis. The FieldMultiVector then
 manages the copying of data between the application and the Tpetra vector
 using the client implementations for data access.
 */
//---------------------------------------------------------------------------//
class LibmeshFieldMultiVector: public Tpetra::MultiVector<double, int, unsigned long int> {
public:

	//! MultiVector typedef.
	typedef Tpetra::MultiVector<double, int, unsigned long int> Base;
	typedef typename Base::local_ordinal_type LO;
	typedef typename Base::global_ordinal_type GO;

	/*!
	 * \brief Comm constructor. This will allocate the Tpetra vector.
	 *
	 * \param field The field for which we are building a vector.
	 *
	 * \param global_comm The global communicator over which the field is
	 * defined.
	 */
	LibmeshFieldMultiVector(
			const Teuchos::RCP<const Teuchos::Comm<int>> &global_comm,
			const Teuchos::RCP<LibmeshVariableField> &field) :
			Base(
					Tpetra::createNonContigMap<int, unsigned long int>(
							field->getLocalSupportIds(), global_comm),
					field->dimension()), d_field(field) { /* ... */
	}

	/*!
	 * \brief Entity set constructor. This will allocate the Tpetra vector.
	 *
	 * \param field The field for which we are building a vector.
	 *
	 * \param entity_set The entity set over which the field is defined.
	 */
	LibmeshFieldMultiVector(const Teuchos::RCP<LibmeshVariableField> &field,
			const Teuchos::RCP<LibmeshEntitySet> &entity_set) :
			Base(
					Tpetra::createNonContigMap<int, unsigned long int>(
							field->getLocalSupportIds(),
							entity_set->communicator()), field->dimension()), d_field(
					field) { /* ... */
	}

	/*!
	 * \brief Pull data from the application and put it in the vector.
	 */
	void pullDataFromApplication() {
		Teuchos::ArrayView<const unsigned long int> field_supports =
				d_field->getLocalSupportIds();

		int num_supports = field_supports.size();
		int dim = d_field->dimension();

		for (int d = 0; d < dim; ++d) {
			Teuchos::ArrayRCP<double> vector_view = this->getDataNonConst(d);
			for (int n = 0; n < num_supports; ++n) {
				vector_view[n] = d_field->readFieldData(field_supports[n], d);
			}
		}
	}

	/*!
	 * \brief Push data from the vector into the application.
	 */
	void pushDataToApplication() {
		Teuchos::ArrayView<const unsigned long int> field_supports =
				d_field->getLocalSupportIds();

		int num_supports = field_supports.size();
		int dim = d_field->dimension();

		for (int d = 0; d < dim; ++d) {
			Teuchos::ArrayRCP<double> vector_view = this->getDataNonConst(d);
			for (int n = 0; n < num_supports; ++n) {
				d_field->writeFieldData(field_supports[n], d, vector_view[n]);
			}
		}

		d_field->finalizeAfterWrite();
	}

private:
	// The field this multivector is managing.
	Teuchos::RCP<LibmeshVariableField> d_field;
};

//---------------------------------------------------------------------------//

}// end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_FIELDMULTIVECTOR_HPP

//---------------------------------------------------------------------------//
// end DTK_FieldMultiVector.hpp
//---------------------------------------------------------------------------//
