# Add CUDA executable
add_executable(state_expand main.cu)

# add flags
target_compile_options(state_expand PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)

# Link against CUDA libraries
target_link_libraries(state_expand ${CUDA_LIBRARIES})

# Set CUDA specific flags
set_target_properties(state_expand PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)

# Add include directories
target_include_directories(state_expand PRIVATE
${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

