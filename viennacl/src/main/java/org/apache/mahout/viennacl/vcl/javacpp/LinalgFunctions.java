package org.apache.mahout.viennacl.vcl.javacpp;

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
