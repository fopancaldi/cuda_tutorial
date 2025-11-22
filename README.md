Code I have used to display some basic functionalities of [CUDA](https://developer.nvidia.com/cuda-toolkit) while learning it.

# Getting started

## Prerequisites

The code should be run on a CUDA-capable machine, i.e. one where the CUDA toolkit was installed.

There are no dependencies.

## Installation

Clone the repo.

# Usage

    cmake -S . -B build
    cmake --build build

Then `build` will contain one executable per source file in `cu`.

# Coding conventions

Use `snake_case` for variables, non-templated types and concepts.
Use `PascalCase` for templated types.

Use `east const`.

Use `and`, `or`, `not` instead of `&&`, `||`, `!`;
