package org.apache.mahout.viennacl.javacpp

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation._

import scala.collection.mutable.ArrayBuffer


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    include = Array("matrix.hpp", "detail/matrix_def.hpp"),
    library = "jniViennaCL"
  )))
@Name(Array("viennacl::matrix_base<double>"))
class MatrixBase extends Pointer {

  protected val ptrs = new ArrayBuffer[Pointer]()

  override def deallocate(deallocate: Boolean): Unit = {
    super.deallocate(deallocate)
    ptrs.foreach(_.close())
  }

  @Name(Array("operator="))
  @native def :=(@Const @ByRef src: DenseRowMatrix)

  @Name(Array("operator="))
  @native def :=(@Const @ByRef src: DenseColumnMatrix)

  @Name(Array("size1"))
  @native
  def nrow: Int

  @Name(Array("size2"))
  @native
  def ncol: Int

  @Name(Array("row_major"))
  @native
  def isRowMajor: Boolean

  @Name(Array("internal_size1"))
  @native
  def internalnrow: Int

  @Name(Array("internal_size2"))
  @native
  def internalncol: Int

  @Name(Array("memory_domain"))
  @native
  def memoryDomain: Int

  @Name(Array("switch_memory_context"))
  @native
  def switchMemoryContext(@ByRef ctx: Context)



}
