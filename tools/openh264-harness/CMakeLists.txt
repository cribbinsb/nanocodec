# openh264-harness: evaluation driver for the openh264 anchor encoder.
#
#   cmake -S tools/openh264-harness -B build/openh264-harness \
#         -DOPENH264_PREFIX=$HOME/nanocodec-data/anchors/$(uname -m)
#   cmake --build build/openh264-harness
#
# The binary is installed next to the anchors so nceval finds it.

cmake_minimum_required(VERSION 3.16)
project(openh264_harness C)

set(OPENH264_PREFIX "" CACHE PATH "Install prefix of the pinned openh264 build")
if(NOT OPENH264_PREFIX)
  message(FATAL_ERROR "set -DOPENH264_PREFIX=<anchors install prefix>")
endif()

find_library(OPENH264_LIB NAMES openh264 PATHS "${OPENH264_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(openh264-harness openh264_harness.c)
set_target_properties(openh264-harness PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
target_compile_options(openh264-harness PRIVATE -Wall -Wextra -Werror -O2)
target_include_directories(openh264-harness PRIVATE "${OPENH264_PREFIX}/include")
# openh264 is C++: when the static libopenh264.a is picked, this C program
# must also pull in the C++ runtime, libm and pthreads.
find_package(Threads REQUIRED)
target_link_libraries(openh264-harness PRIVATE "${OPENH264_LIB}" stdc++ m Threads::Threads)
set_target_properties(openh264-harness PROPERTIES
  BUILD_RPATH "${OPENH264_PREFIX}/lib" INSTALL_RPATH "${OPENH264_PREFIX}/lib")

install(TARGETS openh264-harness DESTINATION "${OPENH264_PREFIX}/bin")
