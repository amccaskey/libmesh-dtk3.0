from dalg24/dtk-stack
run cd $TRILINOS_DIR && git clone --recursive https://github.com/ORNL-CEES/DataTransferKit \
    && cd DataTransferKit && mkdir build && cd build && ../scripts/docker_cmake \
    -DCMAKE_INSTALL_PREFIX=/opt/dtk && make -j4 && make install
run mkdir /cdt-includes && mkdir /cdt-includes/boost \
    && cp -r /opt/boost/1.63.0/include/boost/* /cdt-includes/boost/ \
    && mkdir /cdt-includes/libmesh && cp -r /opt/libmesh/1.2.0/include/* /cdt-includes/ \
    && cp -r /usr/include/* /cdt-includes/ && mkdir /cdt-includes/dtk-trilinos \
    && cp -r /opt/dtk/include /cdt-includes/dtk-trilinos/
run mkdir /projects
workdir /projects
