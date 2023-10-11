# OCR Climate

This is the start of Optical Character Recognition for historical climate data. Basically, handwritten text recognition within the very specific context of tabulated data.

Given the context of this problem, snippets of tables with numbers, it can be approached solving a captcha. With the table cell boundary lines this comparison is more than fitting.

Generally solving captcha problems can be done using an RCNN + CTC loss setup. The Keras introduction into the [captcha problem](https://keras.io/examples/vision/captcha_ocr/) provides the baseline, to be expanded to [handwritten text](https://keras.io/examples/vision/handwriting_recognition/) in another demo. In this quick test I use the vanilla handwritten text recognition code, adapted to the COBECORE climate data formatting (separate labels and images) to learn to recognize the value of climate variables.

A simple test on a subset of the data ~10K images in total (instead of the total dataset of ~350K) shows reasonable performance (see image below). Given these results the exercise should be expanded to the full dataset, including many more writing styles to increase model robustness.

![](https://github.com/khufkens/OCR_climate/blob/main/manuscript/test_results.png)

## Setup

### Getting Tensorflow to work (sort of consistently)

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

### Custom Dockerfile

A docker file is included which builds a custom environment, using the tensorflow setup as a basis, adding Jupyter notebooks for easier interaction with the docker container and the option to visualize things (the other option is forwarding X sessions which is security wise probably iffy).
