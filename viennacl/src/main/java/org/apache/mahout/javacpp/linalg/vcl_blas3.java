package org.apache.mahout.javacpp.linalg;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Platform(includepath={"/usr/include/","/usr/include/CL/","/usr/include/viennacl/"},
 include={"vcl_blas3.h"})
//@Namespace("mmul")
public class vcl_blas3 {
    @Name("dense_dense_mmul")
    public static native void dense_dense_mmul(DoublePointer mxA, long mxANrow, long mxANcol,
                                               DoublePointer mxB, long mxBNrow, long mxBNcol,
                                               @ByRef DoublePointer mxRes);
}
