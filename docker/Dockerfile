from dalg24/dtk-stack
run cd $TRILINOS_DIR && git clone --recursive https://github.com/ORNL-CEES/DataTransferKit \
    && cd DataTransferKit && mkdir build && cd build && ../scripts/docker_cmake \
    -DCMAKE_INSTALL_PREFIX=/opt/dtk && make -j4 && make install
run mkdir /cdt-includes && mkdir /cdt-includes/boost \
    && cp -r /opt/boost/1.62.0/include /cdt-includes/boost/ \
    && mkdir /cdt-includes/libmesh && cp -r /opt/libmesh/1.1.0/include /cdt-includes/libmesh/ \
    && cp -r /usr/include/* /cdt-includes/
