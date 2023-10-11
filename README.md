# OCR Climate

Optical Character Recognition for historical climate data


## Getting Tensorflow to work (sort of consistently)

Install docker per usual and add the user to the
docker user group.

```
sudo usermod -aG docker $USER
```

Install the Nvidia container toolkit.

```
sudo apt install nvidia-container-toolkit
```

Configure the toolkit to recognize the GPUs in docker.

```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Run grab the default image to built upon, this should fire up a console in which the GPUs should be active.

```
docker run --gpus all -it --rm nvcr.io/nvidia/tensorflow:23.09-tf2-py3
```

Mounting external files can be done using:

```
docker run --gpus all -it --rm -v $(pwd):/current_project/ nvcr.io/nvidia/tensorflow:23.09-tf2-py3
```
Which will set /current_project/ as the mount point for the current (present) working directory (pwd).

## Custom Dockerfile

A docker file is included which builds a custom environment, using the tensorflow setup as a basis, adding Jupyter notebooks for easier interaction with the docker container and the option to visualize things (the other option is forwarding X sessions which is security wise probably iffy).





