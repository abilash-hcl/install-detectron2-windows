# Install Detectron2 in Windows platform
Detectron2 is a platform for object detection, segmentation and other visual recognition tasks
https://github.com/facebookresearch/detectron2


## Step-by step instructions on how to install detectron2 in Windows machine without conda

### Platform details
Windows 11 Enterprise
64-bit operating system, x64-based processor

#### Make sure all the software, libraries are fully 64-bit (x64) #
#### Make sure the path to the installation directory for the detectron doesn't contain any spaces (like, "/One Drive/" etc.) #

### Following are the Software, Library details

Clone the Detectron2 from, https://github.com/conansherry/detectron2
 
- Install all the build VS 2022 desktop C++ build tools, also install the community edition
<p align="center">
  <img src="./VS_Build_Tools-2022.png" title="VS_Build_Tools-2019"> 
</p>


- Install Python 3.7.0 (https://www.python.org/downloads/release/python-370/) -  create a new environment
  
- pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

After the Torch is installed, make the changes in some of the package modules (refer https://github.com/conansherry/detectron2), 

File1: 
<p style="color: gray; text-align: center; font-style =  italic; text-indent: 50;"> 
  {your evn path}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\torch\csrc\jit\argument_spec.h(190)
    static constexpr size_t DEPTH_LIMIT = 128;
      change to -->
    static const size_t DEPTH_LIMIT = 128;
 </p>

File2: 
 <p style="color: gray; text-align: center; font-style =  italic; text-indent: 50;"> 
  {your evn path}\Lib\site-packages\torch\include\pybind11\cast.h
  example:
  {C:\Miniconda3\envs\py36}\Lib\site-packages\torch\include\pybind11\cast.h(1449)
    explicit operator type&() { return *(this->value); }
      change to -->
    explicit operator type&() { return *((type*)this->value); }
 </p>

- pip install git+https://github.com/facebookresearch/fvcore
- pip install cython
- pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI 
- pip install setuptools==59.6
- pip install cmake
- pip install ninja
 
Then, open the X64 Native tools command prompt,  load the python environment, go to the detectron repo and run  
- python setup.py build develop

### Thanks #


<p align="center">
  <img src="./Installation_in_X64_NT_cmd.png" title="Installation in X64 NT command prompt">
</p>
