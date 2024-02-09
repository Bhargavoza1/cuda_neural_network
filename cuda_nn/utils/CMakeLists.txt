# CMakeLists.txt in the 'utils' directory

# Glob all source files
file(GLOB SOURCES "*.cpp" "*.cu")

# Add library target
add_library(Utils ${SOURCES})

# Set properties (CUDA language for CUDA files)
set_target_properties(Utils PROPERTIES
    CUDA_STANDARD 20
    CUDA_SEPARABLE_COMPILATION ON
)
