#!/bin/bash
NUMCPUS=`grep -c '^processor' /proc/cpuinfo` # Get max number of CPU threads
NUMCPUS=$(echo "($NUMCPUS*0.75+0.5)/1" | bc) # Set CPU threads to 3/4th avail. threads, rounding to nearest whole number
printf "\033[33;1mMake sure you've installed OpenCL and OpenBLAS by using \"sudo apt install libclblast-dev libopenblas-dev\"\n\n\n\n\n\n\033[0m\n"
sleep 4
# install dependencies
pip install pyinstaller customtkinter && make clean && \
# Ensure all backends are built then build executable file
make LLAMA_HIPBLAS=1 LLAMA_VULKAN=1 LLAMA_OPENBLAS=1 -j$NUMCPUS && \
chmod +x "./create_ver_file.sh"
. create_ver_file.sh
pyinstaller --noconfirm --onefile --clean --console --collect-all customtkinter --collect-all jinja2 --collect-all psutil --icon "./niko.ico" \
--add-data "./kcpp_adapters:./kcpp_adapters" \
--add-data "./koboldcpp.py:." \
--add-data "./json_to_gbnf.py:." \
--add-data "./LICENSE.md:."  \
--add-data "./MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:." \
--add-data "./embd_res:./embd_res" \
--add-data "./koboldcpp_default.so:." \
--add-data "./koboldcpp_hipblas.so:." \
--add-data "/opt/rocm/lib/libhipblas.so:." \
--add-data "/opt/rocm/lib/librocblas.so:." \
--add-data "./koboldcpp_failsafe.so:." \
--add-data "./koboldcpp_noavx2.so:." \
--add-data "./koboldcpp_clblast.so:." \
--add-data "./koboldcpp_clblast_noavx2.so:." \
--add-data "./koboldcpp_clblast_failsafe.so:." \
--add-data "./koboldcpp_vulkan_noavx2.so:." \
--add-data "./koboldcpp_vulkan.so:." \
--add-data "/opt/rocm/lib/rocblas:." \
--version-file "./version.txt" \
"./koboldcpp.py" -n "koboldcpp_rocm"
