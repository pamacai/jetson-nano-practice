To install TensorRT SDK on your host machine, follow these steps:

1. **Download the TensorRT SDK**
    Visit the [NVIDIA TensorRT Download Page](https://developer.nvidia.com/tensorrt) and download the appropriate version for your system. For example, for Ubuntu 20.04, you might download a file like:
    `nv-tensorrt-local-repo-ubuntu2004-10.10.0-cuda-11.8_1.0-1_amd64.deb`

2. **Add the local repository**
    Install the downloaded `.deb` file to add the TensorRT repository to your system:
    ```bash
    sudo dpkg -i nv-tensorrt-local-repo-ubuntu2004-10.10.0-cuda-11.8_1.0-1_amd64.deb
    sudo apt-key add /var/nv-tensorrt-local-repo-*/7fa2af80.pub
    ```

3. **Update the package list**
    Run the following command to update your package list:
    ```bash
    sudo apt-get update
    ```

4. **Install TensorRT packages**
    Install the required TensorRT libraries and development headers:
    ```bash
    sudo apt-get install libnvinfer-dev libnvinfer-headers-dev
    sudo apt-get install libnvinfer-plugin-dev libnvinfer-plugin10
    ```

5. **Verify the installation**
    Check if the TensorRT header file and libraries are installed correctly:
    ```bash
    sudo find / -name NvInfer.h
    find /usr -name "libnvinfer_plugin.so"
    ```

6. **Optional: Install Python bindings**
    If your project requires Python bindings for TensorRT, install them as follows:
    ```bash
    sudo apt-get install python3-libnvinfer python3-libnvinfer-dev
    ```

These steps will set up TensorRT on your host machine, enabling you to build and run CUDA projects that depend on TensorRT.
https://developer.nvidia.com/tensorrt/download/10x
