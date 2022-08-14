FROM dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  hmmer \
  kalign

# Compile HHsuite from source.
RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
  && mkdir /tmp/hh-suite/build \
  && pushd /tmp/hh-suite/build \
  && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
  && make -j 4 && make install \
  && ln -s /opt/hhsuite/bin/* /usr/bin \
  && popd \
  && rm -rf /tmp/hh-suite

RUN ldconfig && \
  apt-get clean && \
  apt-get autoremove && \
  rm -rf /var/lib/apt/lists/* /tmp/* && \
  conda clean -ya
