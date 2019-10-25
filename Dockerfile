FROM vogi23/python36-jupyterlab:1.0.0

LABEL maintainer="Christian von Gunten <chrigu@vgbau.ch>"

ENV MAIN_PATH=/project

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip

RUN pip install -r /tmp/requirements.txt

RUN apt-get install -y git 
RUN git clone https://github.com/GPflow/GPflow.git /tmp/GPflow
RUN pip install /tmp/GPflow  --no-cache-dir

EXPOSE 8888

WORKDIR $MAIN_PATH

ENTRYPOINT ["/bin/bash", "-c", "jupyter lab --no-browser"]