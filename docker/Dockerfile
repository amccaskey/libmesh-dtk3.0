from dalg24/dtk-stack
run cd $TRILINOS_DIR && git clone --recursive https://github.com/ORNL-CEES/DataTransferKit \
    && cd DataTransferKit && mkdir build && cd build && ../scripts/docker_cmake \
    -DCMAKE_INSTALL_PREFIX=/opt/dtk && make -j4 && make install
