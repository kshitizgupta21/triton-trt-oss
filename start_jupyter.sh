#!/bin/bash

# Port at which Jupyter should be run inside the container
PORT=$1

# default port 8321
if [[ -z  $PORT ]]; then
    PORT=8888
fi

jupyter lab --allow-root --ip 0.0.0.0 --port ${PORT} --no-browser --NotebookApp.token='' --NotebookApp.default_url='/lab?reset'
