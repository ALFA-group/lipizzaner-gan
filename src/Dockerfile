FROM nvidia/cuda:9.0-devel
ENV config_file NONE
ENV role NONE
ENV TZ=America/New_York

ENV SWARM False
ENV DOCKER True

WORKDIR /usr/src/app

# Install Python 3.6
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install tzdata software-properties-common curl inetutils-ping -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install python3.6 libpython3.6 python3.6-dev -y

# Install pip
RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" && \
    python3.6 get-pip.py

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY helper_files/requirements.txt ./helper_files/
RUN pip install -r ./helper_files/requirements.txt

COPY . .
CMD [ "sh", "-c", "python3.6 ./main.py train --distributed --${role} -f ${config_file}" ]