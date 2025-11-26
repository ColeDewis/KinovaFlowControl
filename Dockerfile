FROM osrf/ros:noetic-desktop-full
ARG USER=user
ARG DEBIAN_FRONTEND=noninteractive

COPY packages.txt packages.txt

# realsense setup so we can use 405
RUN mkdir -p /etc/apt/keyrings
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list

# install dependencies
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
RUN apt-get update && apt-get install -y \
    $(cat packages.txt) 

COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install  --ignore-installed -r requirements.txt

RUN conan config set general.revisions_enabled=1 && \
    conan profile new default --detect > /dev/null && \
    conan profile update settings.compiler.libcxx=libstdc++11 default
RUN rosdep update

# 3d diffusion setup
RUN apt-get update && apt-get install -y \
    libglew-dev \
    patchelf \
    wget

RUN python3 -m pip install torch torchvision torchaudio
RUN python3 -m pip install --ignore-installed zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
RUN python3 -m pip install --ignore-installed kaleido plotly

RUN mkdir -p ~/.mujoco && \
    cd ~/.mujoco && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate && \
    tar -xvzf mujoco210.tar.gz

ENV MUJOCO_DIR=/root/.mujoco/mujoco210/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_DIR}:/usr/lib/nvidia:/usr/local/cuda/lib64
ENV MUJOCO_GL=egl

RUN git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git
RUN cd 3D-Diffusion-Policy/3D-Diffusion-Policy && \
    pip install .

# 3d-diffusion third party
RUN python3 -m pip uninstall -y cython
RUN sed -i "s/dist.ext_modules = cythonize(\[self.extension\])/dist.ext_modules = cythonize(\[self.extension\], compiler_directives={'legacy_implicit_noexcept': True})/g" 3D-Diffusion-Policy/third_party/mujoco-py-2.1.2.14/mujoco_py/builder.py
RUN python3 -m pip install Cython==0.29.35 -I && \
    cd 3D-Diffusion-Policy/third_party && \
    cd mujoco-py-2.1.2.14 && python3 -m pip install .
RUN cd 3D-Diffusion-Policy/third_party && \
    cd dexart-release && pip install --ignore-installed . && cd .. && \
    cd gym-0.21.0 && pip install --ignore-installed . && cd .. && \
    cd Metaworld && pip install --ignore-installed . && cd .. && \
    cd rrl-dependencies && pip install mj_envs/. && pip install mjrl/. && cd .. 

# 3d-diffusion visualization
RUN cd 3D-Diffusion-Policy/visualizer && \
    pip install .

RUN pip install huggingface_hub=0.25.0

# cuda stuff for torch3d
ARG CUDA_MAJOR_VERSION=12
ARG CUDA_VERSION=12.1
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-1
RUN git clone https://github.com/NVIDIA/cub.git /opt/cub

ENV CUB_HOME=/opt/cub
ENV CUDA_HOME=/usr/local/cuda-$CUDA_VERSION
RUN ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"
RUN cd /3D-Diffusion-Policy/third_party/pytorch3d_simplified/ && python3 -m pip install .

RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc
RUN echo "export MUJOCO_GL=egl" >> ~/.bashrc

# aliases
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo "source /home/user/kinova_flow/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "alias die='tmux kill-server'" >> ~/.bashrc
RUN echo "alias gripper='rosrun kortex_bringup gripper.py'" >> ~/.bashrc
RUN echo "alias kortex_home='rosrun kortex_bringup send_gen3_home.py'" >> ~/.bashrc