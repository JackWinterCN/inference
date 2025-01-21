if [ -d build ]; then
    echo "build dir exists"
    # rm -r build/*
else
    echo "build dir not exists"
    mkdir build
fi
cd build
cmake \
    -DTHIRDPARTY_DIR=/opt/apollo/neo/ \
    -DGPU_PLATFORM=GPU_PLATFORM \
    -DUSE_GPU=1 \
    -DCMAKE_BUILD_TYPE=Debug \
    ../
	# -DPROTOC_DIR=/project/thirdparty/X86_64 \
	# -DPUBLIC_MODULE_DIR=${ZHITO_ROOT}/${OutputDir} \
# cmake ..
cmake --build .
cp inference_demo ../