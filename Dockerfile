FROM ubuntu:latest

WORKDIR ./lipizzaner-gan

RUN apt update
RUN apt install -y python3 python3-pip vim wget

CMD ["/bin/bash"]