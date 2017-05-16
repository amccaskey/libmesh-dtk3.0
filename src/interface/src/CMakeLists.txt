
#
# A) Package-specific configuration options
#

#
# B) Define the header and source files (and directories)
#

SET(HEADERS "")
SET(SOURCES "")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Client)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector)

set(HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_ClientManager.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_Entity.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityExtraData.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityImpl.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityIntegrationRule.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityIterator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityLocalMap.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_Field.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntitySet.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityShapeFunction.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_Types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_BasicEntityPredicates.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_FieldMultiVector.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_FunctionSpace.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_IntegrationPoint.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_IntegrationPointSet.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_MapOperator.hpp
  )

set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_Entity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityIterator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityLocalMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntitySet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Client/DTK_EntityShapeFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_BasicEntityPredicates.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_FieldMultiVector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_FunctionSpace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_IntegrationPointSet.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OperatorVector/DTK_MapOperator.cpp
  )

#
# C) Define the targets for package's library(s)
#

add_library(dtk_interface SHARED ${SOURCES})

