cmake_minimum_required(VERSION 2.8.11) 

set (proj_name cloth-dynamics)

project(${proj_name} C CXX)               

set(CMAKE_CXX_STANDARD 11)  

file(GLOB HEADER_FILES *.h)
file(GLOB SRC_FILES *.c *.cpp)

add_executable(${proj_name} ${SRC_FILES} ${HEADER_FILES})

target_compile_definitions(
  ${proj_name} 
  PRIVATE 
  -DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  -DPROJECT_NAME_STR="${proj_name}")

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(GLFW REQUIRED glfw3)

target_include_directories(
  ${proj_name} 
  PRIVATE
  ${CMAKE_SOURCE_DIR} 
  ${OPENGL_INCLUDE_DIR} 
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/glm 
  ${GLFW_INCLUDE_DIRS})

target_link_libraries(
  ${proj_name} 
  ${OPENGL_gl_LIBRARY} 
  ${GLFW_LIBRARIES} 
  ${CMAKE_DL_LIBS} )


