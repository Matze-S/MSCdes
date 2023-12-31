@del /Q *.lib *.exp *.obj cuda*.exe cuda*64 cuda*32 cuda*.log
@cl >nul 2>&1 /?&if errorlevel 1 call "%ProgramFiles(x86)%\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvars64.bat"
@verify>nul 2>&1

@REM nvcc -o cudaDES64.exe cudaDES.cu --verbose 
@REM --extensible-whole-program --optimize=65535 -maxrregcount=80 
@REM --ftz=true --prec-div=false --prec-sqrt=false --fmad=true --use_fast_math ^
if not errorlevel 0 ^
nvcc ^
--use-local-env --machine 64 --source-in-ptx --generate-line-info --gpu-architecture=sm_75 ^
-Xcudafe="--diag_suppress=1021 --diag_suppress=1031 --diag_suppress=1114 --diag_suppress=1750" ^
-Xcompiler -DWIN32 -x cu -ptx -o cudaDES.ptx cudaDES.cu --keep-device-functions
if not errorlevel 1 ^
nvcc ^
--resource-usage --use-local-env --machine 64 --gpu-architecture=sm_75 -cudart static ^
-Xcudafe="--diag_suppress=1021 --diag_suppress=1031 --diag_suppress=1114 --diag_suppress=1750" ^
-Xcompiler -DWIN32 -x cu -o cudaDES.exe cudaDES.cu --keep-device-functions

if not errorlevel 1 if     "%*""%*"=="""" cudaDES.exe -14     %1 %2 %3 %4 %5 %6 %7 %8 %9
if not errorlevel 1 if     "%*""%*"=="""" cudaDES.exe +17 *18 %1 %2 %3 %4 %5 %6 %7 %8 %9
if not errorlevel 1 if     "%*""%*"=="""" cudaDES.exe -28
@if not errorlevel 1 if     "%*""%*"=="""" echo DONE SUCCESSFULL.
@if not errorlevel 1 if not "%*""%*"=="""" cudaDES.exe         %1 %2 %3 %4 %5 %6 %7 %8 %9

@REM if not errorlevel 1 if     "%*""%*"=="""" cudaDES.exe FE00401EFCFEFEFE %3 %4 %5 %6 %7 %8 %9

@EXIT /B

plaintext : 0123456789ABCDEF
ciphertext: 1A286FAB847AE520
key       : 0000001EFEFEFEFE
StartKey: : FE00401EFCFEFEFE
----------------------------
            FE00400001000000  
            :FC=slice 
            :  FE/2=7F, 
            :  TID=2004=
            :  4+2000/4=
            :  4+500=
            :  504/4=252,,
            :LOOP=1000000:
            :250.505s, :
            :0x400 threads:
            
--------------------------------------------------------------------------------
@REM --compile
@REM -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_37,code=\"sm_37,compute_37\" -gencode=arch=compute_50,code=\"sm_50,compute_50\" -gencode=arch=compute_52,code=\"sm_52,compute_52\" -gencode=arch=compute_60,code=\"sm_60,compute_60\" -gencode=arch=compute_61,code=\"sm_61,compute_61\" -gencode=arch=compute_70,code=\"sm_70,compute_70\" -gencode=arch=compute_75,code=\"sm_75,compute_75\" 
@REM -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\x86_amd64"
@REM -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\/include" -I../../common/inc -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include"  
@REM --warning_suppress_boolean_declared_but_not_referenced --warn_suppress_boolean_declared_but_not_referenced
@REM "/wd 4819" "/Wd 1021" "/Wd 1031" "/EHsc /W1 /nologo /O2 /Fdx64\Release\vc140.pdb /FS /Zi  /MT
--------------------------------------------------------------------------------
Usage  : nvcc [options] <inputfile>

Options for specifying the compilation phase
============================================
More exactly, this option specifies up to which stage the input files must be compiled,
according to the following compilation trajectories for different input file types:
        .c/.cc/.cpp/.cxx : preprocess, compile, link
        .o               : link
        .i/.ii           : compile, link
        .cu              : preprocess, cuda frontend, PTX assemble,
                           merge with host C code, compile, link
        .gpu             : cicc compile into cubin
        .ptx             : PTX assemble into cubin.

--cuda  (-cuda)                         
        Compile all .cu input files to .cu.cpp.ii output.

--cubin (-cubin)                        
        Compile all .cu/.gpu/.ptx input files to device-only .cubin files.  This
        step discards the host code for each .cu input file.

--fatbin(-fatbin)                       
        Compile all .cu/.gpu/.ptx/.cubin input files to device-only .fatbin files.
        This step discards the host code for each .cu input file.

--ptx   (-ptx)                          
        Compile all .cu/.gpu input files to device-only .ptx files.  This step discards
        the host code for each of these input file.

--preprocess                               (-E)                            
        Preprocess all .c/.cc/.cpp/.cxx/.cu input files.

--generate-dependencies                    (-M)                            
        Generate a dependency file that can be included in a make file for the .c/.cc/.cpp/.cxx/.cu
        input file (more than one are not allowed in this mode).

--dependency-output                        (-MF)                           
        Specify the output file for the dependency file generated with -M or -MM.
        If this option is not specified, the output is the same as if -E has been
        specified. 

--compile                                  (-c)                            
        Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file.

--device-c                                 (-dc)                           
        Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
        relocatable device code.  It is equivalent to '--relocatable-device-code=true
        --compile'.

--device-w                                 (-dw)                           
        Compile each .c/.cc/.cpp/.cxx/.cu input file into an object file that contains
        executable device code.  It is equivalent to '--relocatable-device-code=false
        --compile'.

--device-link                              (-dlink)                        
        Link object files with relocatable device code and .ptx/.cubin/.fatbin files
        into an object file with executable device code, which can be passed to the
        host linker.

--link  (-link)                         
        This option specifies the default behavior: compile and link all inputs.

--lib   (-lib)                          
        Compile all inputs into object files (if necessary) and add the results to
        the specified output library file.

--run   (-run)                          
        This option compiles and links all inputs into an executable, and executes
        it.  Or, when the input is a single executable, it is executed without any
        compilation or linking. This step is intended for developers who do not want
        to be bothered with setting the necessary environment variables; these are
        set temporarily by nvcc).


File and path specifications.
=============================

--output-file <file>                       (-o)                            
        Specify name and location of the output file.  Only a single input file is
        allowed when this option is present in nvcc non-linking/archiving mode.

--pre-include <file>,...                   (-include)                      
        Specify header files that must be preincluded during preprocessing.

--library <library>,...                    (-l)                            
        Specify libraries to be used in the linking stage without the library file
        extension.  The libraries are searched for on the library search paths that
        have been specified using option '--library-path'.

--define-macro <def>,...                   (-D)                            
        Specify macro definitions to define for use during preprocessing or compilation.

--undefine-macro <def>,...                 (-U)                            
        Undefine macro definitions during preprocessing or compilation.

--include-path <path>,...                  (-I)                            
        Specify include search paths.

--system-include <path>,...                (-isystem)                      
        Specify system include search paths.

--library-path <path>,...                  (-L)                            
        Specify library search paths.

--output-directory <directory>             (-odir)                         
        Specify the directory of the output file.  This option is intended for letting
        the dependency generation step (see '--generate-dependencies') generate a
        rule that defines the target object file in the proper directory.

--compiler-bindir <path>                   (-ccbin)                        
        Specify the directory in which the host compiler executable resides.  The
        host compiler executable name can be also specified to ensure that the correct
        host compiler is selected.  In addition, driver prefix options ('--input-drive-prefix',
        '--dependency-drive-prefix', or '--drive-prefix') may need to be specified,
        if nvcc is executed in a Cygwin shell or a MinGW shell on Windows.

--cudart{none|shared|static}              (-cudart)                       
        Specify the type of CUDA runtime library to be used: no CUDA runtime library,
        shared/dynamic CUDA runtime library, or static CUDA runtime library.
        Allowed values for this option:  'none','shared','static'.
        Default value:  'static'.

--libdevice-directory <directory>          (-ldir)                         
        Specify the directory that contains the libdevice library files when option
        '--dont-use-profile' is used.  Libdevice library files are located in the
        'nvvm/libdevice' directory in the CUDA toolkit.

--use-local-env                            --use-local-env                 
        Specify whether the environment is already set up for the host compiler.


Options for specifying behavior of compiler/linker.
===================================================

--profile                                  (-pg)                           
        Instrument generated code/executable for use by gprof (Linux only).

--debug (-g)                            
        Generate debug information for host code.

--device-debug                             (-G)                            
        Generate debug information for device code. Turns off all optimizations.
        Don't use for profiling; use -lineinfo instead.

--generate-line-info                       (-lineinfo)                     
        Generate line-number information for device code.

--optimize <level>                         (-O)                            
        Specify optimization level for host code.

--ftemplate-backtrace-limit <limit>        (-ftemplate-backtrace-limit)    
        Set the maximum number of template instantiation notes for a single warning
        or error to <limit>. A value of 0 is allowed, and indicates that no limit
        should be enforced. This value is also passed to the host compiler if it
        provides an equivalent flag.

--ftemplate-depth <limit>                  (-ftemplate-depth)              
        Set the maximum instantiation depth for template classes to <limit>. This
        value is also passed to the host compiler if it provides an equivalent flag.

--shared(-shared)                       
        Generate a shared library during linking.  Use option '--linker-options'
        when other linker options are required for more control.

--x {c|c++|cu}                             (-x)                            
        Explicitly specify the language for the input files, rather than letting
        the compiler choose a default based on the file name suffix.
        Allowed values for this option:  'c','c++','cu'.

--std {c++03|c++11|c++14},...              (-std)                          
        Select a particular C++ dialect.  Note that this flag also turns on the corresponding
        dialect flag for the host compiler.
        Allowed values for this option:  'c++03','c++11','c++14'.

--no-host-device-initializer-list          (-nohdinitlist)                 
        Do not implicitly consider member functions of std::initializer_list as __host__
        __device__ functions.

--no-host-device-move-forward              (-nohdmoveforward)              
        Do not implicitly consider std::move and std::forward as __host__ __device__
        function templates.

--expt-relaxed-constexpr                   (-expt-relaxed-constexpr)       
        Experimental flag: Allow host code to invoke __device__ constexpr functions,
        and device code to invoke __host__ constexpr functions.Note that the behavior
        of this flag may change in future compiler releases.

--expt-extended-lambda                     (-expt-extended-lambda)         
        Experimental flag: Allow __host__, __device__ annotations in lambda declaration.
        Note that the behavior of this flag may change in future compiler releases.

--machine {32|64}                          (-m)                            
        Specify 32 vs 64 bit architecture.
        Allowed values for this option:  32,64.
        Default value:  64.


Options for passing specific phase options
==========================================
These allow for passing options directly to the intended compilation phase.  Using
these, users have the ability to pass options to the lower level compilation tools,
without the need for nvcc to know about each and every such option.

--compiler-options <options>,...           (-Xcompiler)                    
        Specify options directly to the compiler/preprocessor.

--linker-options <options>,...             (-Xlinker)                      
        Specify options directly to the host linker.

--archive-options <options>,...            (-Xarchive)                     
        Specify options directly to library manager.

--ptxas-options <options>,...              (-Xptxas)                       
        Specify options directly to ptxas, the PTX optimizing assembler.

--nvlink-options <options>,...             (-Xnvlink)                      
        Specify options directly to nvlink.


Miscellaneous options for guiding the compiler driver.
======================================================

--dont-use-profile                         (-noprof)                       
        Nvcc uses the nvcc.profiles file for compilation.  When specifying this option,
        the profile file is not used.

--dryrun(-dryrun)                       
        Do not execute the compilation commands generated by nvcc.  Instead, list
        them.

--verbose                                  (-v)                            
        List the compilation commands generated by this compiler driver, but do not
        suppress their execution.

--keep  (-keep)                         
        Keep all intermediate files that are generated during internal compilation
        steps.

--keep-dir <directory>                     (-keep-dir)                     
        Keep all intermediate files that are generated during internal compilation
        steps in this directory.

--save-temps                               (-save-temps)                   
        This option is an alias of '--keep'.

--clean-targets                            (-clean)                        
        This option reverses the behavior of nvcc.  When specified, none of the compilation
        phases will be executed.  Instead, all of the non-temporary files that nvcc
        would otherwise create will be deleted.

--run-args <arguments>,...                 (-run-args)                     
        Used in combination with option --run to specify command line arguments for
        the executable.

--input-drive-prefix <prefix>              (-idp)                          
        On Windows, all command line arguments that refer to file names must be converted
        to the Windows native format before they are passed to pure Windows executables.
        This option specifies how the current development environment represents
        absolute paths.  Use '/cygwin/' as <prefix> for Cygwin build environments,
        and '/' as <prefix> for MinGW.

--dependency-drive-prefix <prefix>         (-ddp)                          
        On Windows, when generating dependency files (see --generate-dependencies),
        all file names must be converted appropriately for the instance of 'make'
        that is used.  Some instances of 'make' have trouble with the colon in absolute
        paths in the native Windows format, which depends on the environment in which
        the 'make' instance has been compiled.  Use '/cygwin/' as <prefix> for a
        Cygwin make, and '/' as <prefix> for MinGW.  Or leave these file names in
        the native Windows format by specifying nothing.

--drive-prefix <prefix>                    (-dp)                           
        Specifies <prefix> as both --input-drive-prefix and --dependency-drive-prefix.

--dependency-target-name <target>          (-MT)                           
        Specify the target name of the generated rule when generating a dependency
        file (see '--generate-dependencies').

--no-align-double                          --no-align-double               
        Specifies that '-malign-double' should not be passed as a compiler argument
        on 32-bit platforms.  WARNING: this makes the ABI incompatible with the cuda's
        kernel ABI for certain 64-bit types.

--no-device-link                           (-nodlink)                      
        Skip the device link step when linking object files.


Options for steering GPU code generation.
=========================================

--gpu-architecture <arch>                  (-arch)                         
        Specify the name of the class of NVIDIA 'virtual' GPU architecture for which
        the CUDA input files must be compiled.
        With the exception as described for the shorthand below, the architecture
        specified with this option must be a 'virtual' architecture (such as compute_50).
        Normally, this option alone does not trigger assembly of the generated PTX
        for a 'real' architecture (that is the role of nvcc option '--gpu-code',
        see below); rather, its purpose is to control preprocessing and compilation
        of the input to PTX.
        For convenience, in case of simple nvcc compilations, the following shorthand
        is supported.  If no value for option '--gpu-code' is specified, then the
        value of this option defaults to the value of '--gpu-architecture'.  In this
        situation, as only exception to the description above, the value specified
        for '--gpu-architecture' may be a 'real' architecture (such as a sm_50),
        in which case nvcc uses the specified 'real' architecture and its closest
        'virtual' architecture as effective architecture values.  For example, 'nvcc
        --gpu-architecture=sm_50' is equivalent to 'nvcc --gpu-architecture=compute_50
        --gpu-code=sm_50,compute_50'.
        Allowed values for this option:  'compute_30','compute_32','compute_35',
        'compute_37','compute_50','compute_52','compute_53','compute_60','compute_61',
        'compute_62','compute_70','compute_72','compute_75','sm_30','sm_32','sm_35',
        'sm_37','sm_50','sm_52','sm_53','sm_60','sm_61','sm_62','sm_70','sm_72',
        'sm_75'.

--gpu-code <code>,...                      (-code)                         
        Specify the name of the NVIDIA GPU to assemble and optimize PTX for.
        nvcc embeds a compiled code image in the resulting executable for each specified
        <code> architecture, which is a true binary load image for each 'real' architecture
        (such as sm_50), and PTX code for the 'virtual' architecture (such as compute_50).
        During runtime, such embedded PTX code is dynamically compiled by the CUDA
        runtime system if no binary load image is found for the 'current' GPU.
        Architectures specified for options '--gpu-architecture' and '--gpu-code'
        may be 'virtual' as well as 'real', but the <code> architectures must be
        compatible with the <arch> architecture.  When the '--gpu-code' option is
        used, the value for the '--gpu-architecture' option must be a 'virtual' PTX
        architecture.
        For instance, '--gpu-architecture=compute_35' is not compatible with '--gpu-code=sm_30',
        because the earlier compilation stages will assume the availability of 'compute_35'
        features that are not present on 'sm_30'.
        Allowed values for this option:  'compute_30','compute_32','compute_35',
        'compute_37','compute_50','compute_52','compute_53','compute_60','compute_61',
        'compute_62','compute_70','compute_72','compute_75','sm_30','sm_32','sm_35',
        'sm_37','sm_50','sm_52','sm_53','sm_60','sm_61','sm_62','sm_70','sm_72',
        'sm_75'.

--generate-code <specification>,...        (-gencode)                      
        This option provides a generalization of the '--gpu-architecture=<arch> --gpu-code=<code>,
        ...' option combination for specifying nvcc behavior with respect to code
        generation.  Where use of the previous options generates code for different
        'real' architectures with the PTX for the same 'virtual' architecture, option
        '--generate-code' allows multiple PTX generations for different 'virtual'
        architectures.  In fact, '--gpu-architecture=<arch> --gpu-code=<code>,
        ...' is equivalent to '--generate-code arch=<arch>,code=<code>,...'.
        '--generate-code' options may be repeated for different virtual architectures.
        Allowed keywords for this option:  'arch','code'.

--relocatable-device-code {true|false}     (-rdc)                          
        Enable (disable) the generation of relocatable device code.  If disabled,
        executable device code is generated.  Relocatable device code must be linked
        before it can be executed.
        Default value:  false.

--entries entry,...                        (-e)                            
        Specify the global entry functions for which code must be generated.  By
        default, code will be generated for all entry functions.

--maxrregcount <amount>                    (-maxrregcount)                 
        Specify the maximum amount of registers that GPU functions can use.
        Until a function-specific limit, a higher value will generally increase the
        performance of individual GPU threads that execute this function.  However,
        because thread registers are allocated from a global register pool on each
        GPU, a higher value of this option will also reduce the maximum thread block
        size, thereby reducing the amount of thread parallelism.  Hence, a good maxrregcount
        value is the result of a trade-off.
        If this option is not specified, then no maximum is assumed.
        Value less than the minimum registers required by ABI will be bumped up by
        the compiler to ABI minimum limit.
        User program may not be able to make use of all registers as some registers
        are reserved by compiler.

--use_fast_math                            (-use_fast_math)                
        Make use of fast math library.  '--use_fast_math' implies '--ftz=true --prec-div=false
        --prec-sqrt=false --fmad=true'.

--ftz {true|false}                         (-ftz)                          
        This option controls single-precision denormals support. '--ftz=true' flushes
        denormal values to zero and '--ftz=false' preserves denormal values. '--use_fast_math'
        implies '--ftz=true'.
        Default value:  false.

--prec-div {true|false}                    (-prec-div)                     
        This option controls single-precision floating-point division and reciprocals.
        '--prec-div=true' enables the IEEE round-to-nearest mode and '--prec-div=false'
        enables the fast approximation mode.  '--use_fast_math' implies '--prec-div=false'.
        Default value:  true.

--prec-sqrt {true|false}                   (-prec-sqrt)                    
        This option controls single-precision floating-point squre root.  '--prec-sqrt=true'
        enables the IEEE round-to-nearest mode and '--prec-sqrt=false' enables the
        fast approximation mode.  '--use_fast_math' implies '--prec-sqrt=false'.
        Default value:  true.

--fmad {true|false}                        (-fmad)                         
        This option enables (disables) the contraction of floating-point multiplies
        and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA,
        or DFMA).  '--use_fast_math' implies '--fmad=true'.
        Default value:  true.


Options for steering cuda compilation.
======================================

--default-stream {legacy|null|per-thread}  (-default-stream)               
        Specify the stream that CUDA commands from the compiled program will be sent
        to by default.
                
        legacy
                The CUDA legacy stream (per context, implicitly synchronizes with
                other streams).
                
        per-thread
                A normal CUDA stream (per thread, does not implicitly
                synchronize with other streams).
                
        'null' is a deprecated alias for 'legacy'.
                
        Allowed values for this option:  'legacy','null','per-thread'.
        Default value:  'legacy'.


Generic tool options.
=====================

--disable-warnings                         (-w)                            
        Inhibit all warning messages.

--keep-device-functions                    (-keep-device-functions)        
        In whole program compilation mode, preserve user defined external linkage
        __device__ function definitions up to PTX.

--source-in-ptx                            (-src-in-ptx)                   
        Interleave source in PTX. May only be used in conjunction with --device-debug
        or --generate-line-info.

--restrict                                 (-restrict)                     
        Programmer assertion that all kernel pointer parameters are restrict pointers.

--Wreorder                                 (-Wreorder)                     
        Generate warnings when member initializers are reordered.

--Wno-deprecated-declarations              (-Wno-deprecated-declarations)  
        Suppress warning on use of deprecated entity.

--Wno-deprecated-gpu-targets               (-Wno-deprecated-gpu-targets)   
        Suppress warnings about deprecated GPU target architectures.

--Werror<kind>,...                        (-Werror)                       
        Make warnings of the specified kinds into errors.  The following is the list
        of warning kinds accepted by this option:
                
        cross-execution-space-call
                Be more strict about unsupported cross execution space calls.
                The compiler will generate an error instead of a warning for a
                call from a __host__ __device__ to a __host__ function.
        reorder
                Generate errors when member initializers are reordered.
        deprecated-declarations
                Generate error on use of a deprecated entity.
        Allowed values for this option:  'cross-execution-space-call','deprecated-declarations',
        'reorder'.

--resource-usage                           (-res-usage)                    
        Show resource usage such as registers and memory of the GPU code.
        This option implies '--nvlink-options --verbose' when '--relocatable-device-code=true'
        is set.  Otherwise, it implies '--ptxas-options --verbose'.

--extensible-whole-program                 (-ewp)                          
        Do extensible whole program compilation of device code.

--no-compress                              (-no-compress)                  
        Do not compress device code in fatbinary.

--help  (-h)                            
        Print this help information on this tool.

--version                                  (-V)                            
        Print version information on this tool.

--options-file <file>,...                  (-optf)                         
        Include command line options from specified file.

