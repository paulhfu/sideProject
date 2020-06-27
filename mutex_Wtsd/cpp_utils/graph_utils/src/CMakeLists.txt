add_subdirectory(lib)
#
#
##################################
## INSTALL THE PYTHON MODULE
##################################
#IF(NOT DEFINED PYTHON_MODULE_INSTALL_DIR OR PYTHON_MODULE_INSTALL_DIR MATCHES "^$")
#
#
#    execute_process(
#      COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
#        from __future__ import print_function
#        from distutils import sysconfig as sc
#        print(sc.get_python_lib(prefix='', plat_specific=True))"
#      OUTPUT_VARIABLE PYTHON_SITE
#      OUTPUT_STRIP_TRAILING_WHITESPACE)
#
#    execute_process(
#      COMMAND "${PYTHON_EXECUTABLE}" -c "from __future__ import print_function; from distutils.sysconfig import get_python_lib; print(get_python_lib())"
#      OUTPUT_VARIABLE PYTHON_SITE
#      OUTPUT_STRIP_TRAILING_WHITESPACE)
#
#    SET(PYTHON_MODULE_INSTALL_DIR ${PYTHON_SITE})
#
#ENDIF()
#
#SET(PYTHON_MODULE_INSTALL_DIR ${PYTHON_MODULE_INSTALL_DIR}
#    CACHE PATH "where to install the rag_utils python package" FORCE)
#
## this is the install path relative to CMAKE_INSTALL_PREFIX,
## use this in INSTALL() commands to get packaging right
#install(DIRECTORY ${CMAKE_BINARY_DIR}
#	DESTINATION "${CMAKE_PREFIX_PATH}/${PYTHON_MODULE_INSTALL_DIR}")
