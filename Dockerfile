FROM python:3.11 as base
RUN apt-get update && apt-get -y upgrade && apt-get -y install bash git gcc jq
RUN apt-get -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

FROM base as build
WORKDIR /
COPY requirements.txt .
COPY src/. .
RUN pip install -r requirements.txt

FROM build as runtime
CMD ["python", "main.py"]

