# set base docker image to work from
# Tensorflow in this case
FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3

# set workdir
WORKDIR /workspace

# install required python packages
RUN pip install jupyter
RUN pip install keras
RUN pip install matplotlib

# expose outgoing traffic on port 8888
EXPOSE 8888

# In the project run:
#
# docker build -t tensorflow-container .
# docker run --runtime=nvidia -it -p "8888:8888" -v $(pwd):/current_project/ tensorflow-container
#
# Start Jupyter notebooks:
# jupyter notebook -port=8888 --ip=0.0.0.0 --allow-root --no-browser .
# browse to http://localhost:8888 to access the notebook
