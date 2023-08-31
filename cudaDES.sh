#!/bin/bash
rm *.lib *.exp *.obj cudaDES cudaDES*.exe cuda*64 cuda*32 cuda*.log -f

# --ftz=true --prec-div=false --prec-sqrt=false --fmad=true --fast-math --use_fast_math
# --extensible-whole-program --optimize=65535 -maxrregcount=0 
#nvcc --use-local-env --machine 64 --source-in-ptx --generate-line-info --gpu-architecture=sm_70 \
#    -Xcudafe="--diag_suppress=1021 --diag_suppress=1031 --diag_suppress=1114 --diag_suppress=1120" \
#    -Xcompiler -DLINUX -x cu -ptx -o cudaDES.ptx cudaDES.cu --keep-device-functions
nvcc --machine 64 --resource-usage --use-local-env -cudart static --gpu-architecture=sm_70 \
    -Xcudafe="--diag_suppress=1021 --diag_suppress=1031 --diag_suppress=1114 --diag_suppress=1120" \
    -Xcompiler -DLINUX -x cu -o cudaDES cudaDES.cu --keep-device-functions && \
( [ ! -z "$@" ] || ( ./cudaDES "-14" $@ && ./cudaDES "+17" "*18" $@ || ./cudaDES -28 && echo DONE SUCCESSFULL. ) ) && \
( [   -z "$@" ] || ( ./cudaDES $@ ) )
 
