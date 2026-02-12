cmake_minimum_required(VERSION 3.25)
project(mlx_ruby_native LANGUAGES C)

# Phase 1 scaffold: this target keeps CMake wiring in place while extconf/mkmf
# drives the local extension build.
add_library(mlx_ruby_native MODULE native.cpp)
set_target_properties(mlx_ruby_native PROPERTIES PREFIX "")
