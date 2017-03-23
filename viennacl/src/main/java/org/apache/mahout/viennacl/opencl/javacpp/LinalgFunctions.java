/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.viennacl.opencl.javacpp;

import org.bytedeco.javacpp.annotation.*;


@Properties(inherit = Context.class,
        value = @Platform(
                library = "jniViennaCL"
        )
)
@Namespace("viennacl::linalg")
public final class LinalgFunctions {

    private LinalgFunctions() {
    }

    static {
        Context.loadLib();
    }


    @ByVal
    public static native MatMatProdExpression prod(@Const @ByRef MatrixBase a,
                                                   @Const @ByRef MatrixBase b);

    @ByVal
    public static native ProdExpression prod(@Const @ByRef CompressedMatrix a,
                                             @Const @ByRef CompressedMatrix b);

    @ByVal
    public static native MatVecProdExpression prod(@Const @ByRef MatrixBase a,
                                                   @Const @ByRef VectorBase b);

    @ByVal
    public static native SrMatDnMatProdExpression prod(@Const @ByRef CompressedMatrix spMx,
                                                       @Const @ByRef MatrixBase dMx);
    @ByVal
    @Name("prod")
    public static native DenseColumnMatrix prodCm(@Const @ByRef MatrixBase a,
                                                  @Const @ByRef MatrixBase b);
    @ByVal
    @Name("prod")
    public static native DenseRowMatrix prodRm(@Const @ByRef MatrixBase a,
                                               @Const @ByRef MatrixBase b);

    @ByVal
    @Name("prod")
    public static native DenseRowMatrix prodRm(@Const @ByRef CompressedMatrix spMx,
                                               @Const @ByRef MatrixBase dMx);


//    @ByVal
//    public static native MatrixProdExpression prod(@Const @ByRef DenseRowMatrix a,
//                                                   @Const @ByRef DenseRowMatrix b);
//
//    @ByVal
//    public static native MatrixProdExpression prod(@Const @ByRef DenseRowMatrix a,
//                                                   @Const @ByRef DenseColumnMatrix b);
//
//    @ByVal
//    public static native MatrixProdExpression prod(@Const @ByRef DenseColumnMatrix a,
//                                                   @Const @ByRef DenseRowMatrix b);
//
//    @ByVal
//    public static native MatrixProdExpression prod(@Const @ByRef DenseColumnMatrix a,
//                                                   @Const @ByRef DenseColumnMatrix b);


}
