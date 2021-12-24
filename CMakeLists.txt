cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 11)

project(sample)
enable_language(ASM)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O1 -Wall")

add_executable(sample main.cc timelog.h ArmAsmGlobal.h MatMulPack12.S MatMulPack24.S)

