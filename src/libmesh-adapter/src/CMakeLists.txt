INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/interface/src/Client)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/interface/src/OperatorVector)

set(HEADERS
  DTK_LibmeshAdjacencies.hpp
  DTK_LibmeshEntity.hpp
  DTK_LibmeshEntityExtraData.hpp
  DTK_LibmeshEntityImpl.hpp
  DTK_LibmeshEntityIntegrationRule.hpp
  DTK_LibmeshEntityIterator.hpp
  DTK_LibmeshEntityIterator_impl.hpp
  DTK_LibmeshEntityLocalMap.hpp
  DTK_LibmeshEntitySet.hpp
  DTK_LibmeshHelpers.hpp
  DTK_LibmeshManager.hpp
  DTK_LibmeshNodalShapeFunction.hpp
  DTK_LibmeshVariableField.hpp
  )

set(SOURCES
  DTK_LibmeshAdjacencies.cpp
  DTK_LibmeshEntity.cpp
  DTK_LibmeshEntityImpl.cpp
  DTK_LibmeshEntityIntegrationRule.cpp
  DTK_LibmeshEntityLocalMap.cpp
  DTK_LibmeshEntitySet.cpp
  DTK_LibmeshManager.cpp
  DTK_LibmeshNodalShapeFunction.cpp
  DTK_LibmeshVariableField.cpp
  )

add_library(dtk_libmeshadapters SHARED ${SOURCES})
