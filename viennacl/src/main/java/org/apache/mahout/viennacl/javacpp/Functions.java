package org.apache.mahout.viennacl.javacpp;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;


@Properties(inherit = Context.class,
        value = @Platform(
                library = "jniViennaCL"
        )
)
@Namespace("viennacl")
public final class Functions {

    private Functions() {
    }

    // This is (imo) an inconsistency in Vienna cl: almost all operations require MatrixBase, and
    // fast_copy require type `matrix`, i.e., one of DenseRowMatrix or DenseColumnMatrix.
    @Name("fast_copy")
    public static native void fastCopy(DoublePointer srcBegin, DoublePointer srcEnd, @ByRef DenseRowMatrix dst);

    @Name("fast_copy")
    public static native void fastCopy(DoublePointer srcBegin, DoublePointer srcEnd, @ByRef DenseColumnMatrix dst);

    @Name("fast_copy")
    public static native void fastCopy(@ByRef DenseRowMatrix src, DoublePointer dst);

    @Name("fast_copy")
    public static native void fastCopy(@ByRef DenseColumnMatrix src, DoublePointer dst);

    @Name("fast_copy")
    public static native void fastCopy(@Const @ByRef VectorBase dst, @Const @ByRef VCLVector src);

    @Name("fast_copy")
    public static native void fastCopy(@Const @ByRef VCLVector src, @Const @ByRef VectorBase dst);


    @ByVal
    public static native MatrixTransExpression trans(@ByRef MatrixBase src);

    @Name("backend::memory_read")
    public static native void memoryReadInt(@Const @ByRef MemHandle src_buffer,
                                  int bytes_to_read,
                                  int offset,
                                  IntPointer ptr,
                                  boolean async);

    @Name("backend::memory_read")
    public static native void memoryReadDouble(@Const @ByRef MemHandle src_buffer,
                                            int bytes_to_read,
                                            int offset,
                                            DoublePointer ptr,
                                            boolean async);

    @Name("backend::memory_read")
    public static native void memoryReadInt(@Const @ByRef MemHandle src_buffer,
                                            int bytes_to_read,
                                            int offset,
                                            IntBuffer ptr,
                                            boolean async);

    @Name("backend::memory_read")
    public static native void memoryReadDouble(@Const @ByRef MemHandle src_buffer,
                                               int bytes_to_read,
                                               int offset,
                                               DoubleBuffer ptr,
                                               boolean async);

    @Name("backend::memory_read")
    public static native void memoryReadBytes(@Const @ByRef MemHandle src_buffer,
                                              int bytes_to_read,
                                              int offset,
                                              BytePointer ptr,
                                              boolean async);


    static {
        Context.loadLib();
    }

}
