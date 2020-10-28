# Platform-specific dylib extension
if [ $(uname) == "Darwin" ]; then
    export CC=clang
    export CXX=clang++
    export DYLIB="dylib"
else
    export CC=gcc
    export CXX=g++
    export DYLIB="so"
fi

##
## START THE BUILD
##

mkdir -p build
cd build

PREFIX="${CONDA_PREFIX}"
SRC_DIR=".."
PYTHON=$(which python)

CXXFLAGS="${CXXFLAGS} -I${PREFIX}/include"
LDFLAGS="${LDFLAGS} -Wl,-rpath,${PREFIX}/lib -L${PREFIX}/lib"

if [ $(uname) == Darwin ]; then
    CXXFLAGS="$CXXFLAGS -stdlib=libc++"
fi

# Set the correct python flags, depending on the distribution
PY_VER=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))")
PY_ABIFLAGS=$(python -c "import sys; print('' if sys.version_info.major == 2 else sys.abiflags)")
PY_ABI=${PY_VER}${PY_ABIFLAGS}

##
## Configure
##
cmake .. \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=10.9\
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_PREFIX_PATH=${PREFIX} \
\
        -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}" \
        -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
        -DCMAKE_CXX_FLAGS_RELEASE="${CXXFLAGS} -O3 -DNDEBUG" \
        -DCMAKE_CXX_FLAGS_DEBUG="${CXXFLAGS}" \
\
        -DBOOST_ROOT=${PREFIX} \
\
        -DPYTHON_EXECUTABLE=${PYTHON} \
        -DPYTHON_LIBRARY=${PREFIX}/lib/libpython${PY_ABI}.${DYLIB} \
        -DPYTHON_INCLUDE_DIR=${PREFIX}/include/python${PY_ABI} \
##

##
## Compile
##
make -j${CPU_COUNT}
##
## Install to prefix
#mkdir ${PREFIX}/lib/python${PY_VER}/site-packages/rag_utils
echo ${SRC_DIR}/build/python/*
echo ${PREFIX}/lib/python${PY_VER}/site-packages/
cp -r ${SRC_DIR}/build/python/* ${PREFIX}/lib/python${PY_VER}/site-packages/
