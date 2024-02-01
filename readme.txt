1)
cmake -B build . -DCMAKE_INSTALL_PREFIX=install -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release

2)
修改 ./3rd/FreeGLUT/src/fg_gl2.h

FGH_PFNGLGENBUFFERSPROC fghGenBuffers;                             -> static FGH_PFNGLGENBUFFERSPROC fghGenBuffers;
FGH_PFNGLDELETEBUFFERSPROC fghDeleteBuffers;                       -> static FGH_PFNGLDELETEBUFFERSPROC fghDeleteBuffers;
FGH_PFNGLBINDBUFFERPROC fghBindBuffer;                             -> static FGH_PFNGLBINDBUFFERPROC fghBindBuffer;
FGH_PFNGLBUFFERDATAPROC fghBufferData;                             -> static FGH_PFNGLBUFFERDATAPROC fghBufferData;
FGH_PFNGLENABLEVERTEXATTRIBARRAYPROC fghEnableVertexAttribArray;   -> static FGH_PFNGLENABLEVERTEXATTRIBARRAYPROC fghEnableVertexAttribArray;
FGH_PFNGLDISABLEVERTEXATTRIBARRAYPROC fghDisableVertexAttribArray; -> static FGH_PFNGLDISABLEVERTEXATTRIBARRAYPROC fghDisableVertexAttribArray;
FGH_PFNGLVERTEXATTRIBPOINTERPROC fghVertexAttribPointer;           -> static FGH_PFNGLVERTEXATTRIBPOINTERPROC fghVertexAttribPointer;

3)
VERBOSE=1 cmake --build build -j32

4)
./build/simulator


