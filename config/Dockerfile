FROM vogi23/python37:1.0.1

ENV MAIN_PATH=/usr/local/bin/host_dir
ENV CONFIG_PATH=${MAIN_PATH}/config
ENV PROJECT_PATH=${MAIN_PATH}/project

RUN pip install --upgrade pip

#RUN pip install --upgrade <your-packages>

EXPOSE 8888

CMD ["/bin/bash", "-c", "${CONFIG_PATH}/run_jupyter.sh"]