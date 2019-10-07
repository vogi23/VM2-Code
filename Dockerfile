FROM vogi23/python36-jupyterlab:1.0.0

LABEL maintainer="Christian von Gunten <chrigu@vgbau.ch>"

ENV MAIN_PATH=/project

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

EXPOSE 8888

WORKDIR $MAIN_PATH

ENTRYPOINT ["/bin/bash", "-c", "jupyter lab"]