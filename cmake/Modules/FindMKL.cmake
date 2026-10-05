# a simple cmake script to locate Intel Math Kernel Library via MKLROOT
# or fallback to OpenBLAS / Accelerate on platforms without MKL (e.g. macOS Apple Silicon)

# Stage 1: find the root directory

set(MKLROOT_PATH $ENV{MKLROOT})

# Stage 2: find include path and libraries

if (MKLROOT_PATH)
  # includes
  set(EXPECT_MKL_INCPATH "${MKLROOT_PATH}/include")

  if (IS_DIRECTORY ${EXPECT_MKL_INCPATH})
    set(MKL_INCLUDE_DIRS ${EXPECT_MKL_INCPATH})
  endif (IS_DIRECTORY ${EXPECT_MKL_INCPATH})

  # libs
  if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib")
  endif (CMAKE_SYSTEM_NAME MATCHES "Darwin")

  if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(EXPECT_MKL_LIBPATH "${MKLROOT_PATH}/lib/intel64")
  endif (CMAKE_SYSTEM_NAME MATCHES "Linux")

  if (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})
    set(MKL_LIBRARY_DIR ${EXPECT_MKL_LIBPATH})
  endif (IS_DIRECTORY ${EXPECT_MKL_LIBPATH})

  string(REPLACE ":" ";" ICC_LIBRARY_DIR $ENV{LIBRARY_PATH})

  # find specific library files
  find_library(LIB_MKL_CORE NAMES mkl_core HINTS ${MKL_LIBRARY_DIR})
  find_library(LIB_MKL_INTEL_THREAD NAMES mkl_intel_thread
    HINTS ${MKL_LIBRARY_DIR})
  find_library(LIB_MKL_INTEL_ILP64 NAMES mkl_intel_ilp64
    HINTS ${MKL_LIBRARY_DIR})
  find_library(LIB_IOMP5 NAMES iomp5 HINTS ${ICC_LIBRARY_DIR})
  find_library(LIB_PTHREAD NAMES pthread)

  if (CMAKE_SYSTEM_NAME MATCHES Linux AND CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(NO_AS_NEEDED -Wl,--no-as-needed)
  endif (CMAKE_SYSTEM_NAME MATCHES Linux AND CMAKE_CXX_COMPILER_ID MATCHES GNU)

  set(MKL_LIBRARIES
    ${NO_AS_NEEDED}
    ${LIB_MKL_CORE}
    ${LIB_MKL_INTEL_THREAD}
    ${LIB_MKL_INTEL_ILP64}
    ${LIB_IOMP5}
    ${LIB_PTHREAD})

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(MKL DEFAULT_MSG
    MKL_LIBRARY_DIR
    LIB_MKL_CORE
    LIB_MKL_INTEL_THREAD
    LIB_MKL_INTEL_ILP64
    LIB_IOMP5
    LIB_PTHREAD
    MKL_INCLUDE_DIRS)

else()
  # Fallback to OpenBLAS on platforms without MKL (such as macOS ARM64)
  find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h lapacke.h
    PATHS
      /opt/homebrew/opt/openblas/include
      /usr/local/opt/openblas/include
    NO_DEFAULT_PATH
  )

  if (NOT OPENBLAS_INCLUDE_DIR)
    find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h
      PATHS
        /opt/homebrew/include
        /usr/local/include
    )
  endif()

  find_library(LIB_OPENBLAS NAMES openblas
    PATHS
      /opt/homebrew/opt/openblas/lib
      /usr/local/opt/openblas/lib
    NO_DEFAULT_PATH
  )

  if (NOT LIB_OPENBLAS)
    find_library(LIB_OPENBLAS NAMES openblas
      PATHS
        /opt/homebrew/lib
        /usr/local/lib
    )
  endif()

  find_library(LIB_PTHREAD NAMES pthread)

  if (LIB_OPENBLAS AND OPENBLAS_INCLUDE_DIR)
    message(STATUS "MKL not found, using OpenBLAS fallback: ${LIB_OPENBLAS}")
    message(STATUS "OpenBLAS include dir: ${OPENBLAS_INCLUDE_DIR}")
    set(MKL_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/compat/mkl
      ${OPENBLAS_INCLUDE_DIR}
    )
    set(MKL_LIBRARIES ${LIB_OPENBLAS} ${LIB_PTHREAD})
    set(MKL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/compat/mkl/mkl_rci.cpp)
    set(MKL_FOUND TRUE)
  else()
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(MKL DEFAULT_MSG
      MKLROOT_PATH)
  endif()
endif()
