package org.apache.mahout.viennacl.openmp.javacpp

import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.javacpp.annotation._

/**
  * ViennaCL dense matrix, row-major.
  */
@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    library = "jniViennaCL"
  )))
@Name(Array("viennacl::matrix<double,viennacl::row_major>"))
class DenseRowMatrix(initDefault: Boolean = true) extends MatrixBase {

  def this(nrow: Int, ncol: Int) {
    this(false)
    allocate(nrow, ncol, new Context())
  }

  def this(nrow: Int, ncol: Int, ctx: Context) {
    this(false)
    allocate(nrow, ncol, ctx)
  }

  def this(data: DoublePointer, nrow: Int, ncol: Int) {
    this(false)
    allocate(data, new Context(Context.MAIN_MEMORY).memoryType, nrow, ncol)
    // We save it to deallocate it ad deallocation time.
    ptrs += data
  }

  def this(data: DoublePointer, nrow: Int, ncol: Int, ctx: Context) {
    this(false)
    allocate(data, ctx.memoryType, nrow, ncol)
    // We save it to deallocate it ad deallocation time.
    ptrs += data
  }

  def this(me: MatMatProdExpression) {
    this(false)
    allocate(me)
  }

  def this(me: MatrixTransExpression) {
    this(false)
    allocate(me)
  }

  // TODO: getting compilation errors here
  def this(sd: SrMatDnMatProdExpression) {
    this(false)
    allocate(sd)
  }

  if (initDefault) allocate()

  @native protected def allocate()

  @native protected def allocate(nrow: Int, ncol: Int, @ByVal ctx: Context)

  @native protected def allocate(data: DoublePointer,
                                 @Cast(Array("viennacl::memory_types"))
                                 memType: Int,
                                 nrow: Int,
                                 ncol: Int
                                )

  @native protected def allocate(@Const @ByRef me: MatMatProdExpression)

  @native protected def allocate(@Const @ByRef me: MatrixTransExpression)

  // TODO: Compilation errors here
  @native protected def allocate(@Const @ByRef me: SrMatDnMatProdExpression)

}


object DenseRowMatrix {
  Context.loadLib()
}