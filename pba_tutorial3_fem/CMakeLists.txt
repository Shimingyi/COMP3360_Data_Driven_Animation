cmake_minimum_required( VERSION 3.1 )

set (proj_name fem-tutorial)

project(${proj_name} C CXX)               

set(CMAKE_CXX_STANDARD 11)  

if( MSVC )
    SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup" )
endif()

set(CMAKE_CXX_STANDARD 11)  

file(GLOB HEADER_FILES *.h)
file(GLOB SRC_FILES *.c *.cpp)

add_executable(${proj_name} WIN32 ${SRC_FILES} ${HEADER_FILES})

target_compile_definitions(
  ${proj_name} 
  PRIVATE 
  -DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  -DPROJECT_NAME_STR="${proj_name}")

find_package(OpenGL REQUIRED)

set( GLFW_BUILD_DOCS OFF CACHE BOOL  "GLFW lib only" )
set( GLFW_INSTALL OFF CACHE BOOL  "GLFW lib only" )
set( GLFW_BUILD_EXAMPLES OFF CACHE BOOL  "GLFW lib only" )
set( GLFW_BUILD_TESTS OFF CACHE BOOL  "GLFW lib only" )

add_subdirectory(extern/glfw)

target_include_directories(
  ${proj_name} 
  PRIVATE
  ${CMAKE_SOURCE_DIR} 
  ${OPENGL_INCLUDE_DIR} 
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/glm 
  ${CMAKE_CURRENT_SOURCE_DIR}/extern 
  ${GLFW_INCLUDE_DIRS})

target_link_libraries(
  ${proj_name} 
  ${OPENGL_gl_LIBRARY} 
  glfw 
  ${CMAKE_DL_LIBS} )

if( MSVC )
    if(${CMAKE_VERSION} VERSION_LESS "3.6.0") 
        message( "\n\t[ WARNING ]\n\n\tCMake version lower than 3.6.\n\n\t - Please update CMake and rerun; OR\n\t - Manually set '${proj_name} ' as StartUp Project in Visual Studio.\n" )
    else()
        set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${proj_name}  )
    endif()
endif()

