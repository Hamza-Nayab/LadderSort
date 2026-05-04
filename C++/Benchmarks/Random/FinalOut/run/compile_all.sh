#!/usr/bin/env bash
set -e

CXX=/opt/homebrew/bin/g++-15
CXXFLAGS="-O3 -march=native -flto -funroll-loops -std=c++20"
INCLUDES="-I../third_party/cpp-TimSort/include -Iinclude"

mkdir -p bin

$CXX $CXXFLAGS $INCLUDES bench_laddersort_ablation.cpp -o bin/bench_laddersort_ablation
$CXX $CXXFLAGS $INCLUDES bench_postinsert_multi.cpp -o bin/bench_postinsert_multi
$CXX $CXXFLAGS $INCLUDES bench_main_runtime.cpp -o bin/bench_main_runtime