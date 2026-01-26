---
title: Weekly Meeting Minutes 2020-12-08
date: 2020-12-08
tags: [minutes]
authors: [mahout-team]
---

New Meeting time is Tuesdays, still hammering down final time.

Musselman to continue work on 2130 (adding talks page).

Trevor making (slow) progress wtih Py4J / Numpy / Python bindings (life getting in the way, but seems doable).

Palumbo interested in FPGA-
> Investigation: FPGA4Mahout, Mahout BLAS Subroutines  on FPGA.  we are considering integration of the fBLAS library for FPGA acceleration of BLAS subroutines on mahout.  fBLAS accepts JSON expressions of algebraic subroutines and generates OpenCL code with Just In Time compilation and FPGA flashing.  We are first considering a Naive approach upon evaluation to convert the computation graph of a mahout expression into JSON, compile and Flash the FPGA and stream data through the FPGA in SIMD vectored fashion.
>
> The fBLAS library shows significant speedups over CPU BLAS operations.  If we find significant performance gains,  In conjunction with the current and ongoing effort to implement Python Bindings for Mahout, this could attract HPC developers with needs for near-real-time computation, by providing Python and Java bindings for FPGA accelerated matrix algebra available for distributed and in-core math.
>
> An end goal of this effort would be to provide scientists,  engineers, and others with little to no Hardware experience a Zeppelin Notebook on which they could  can develop models or circuits, using the mahout DSL, while concurrntly programming the model onto an FPGA"
