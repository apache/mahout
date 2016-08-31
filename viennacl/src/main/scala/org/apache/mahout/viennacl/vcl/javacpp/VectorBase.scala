package org.apache.mahout.viennacl.vcl.javacpp

import java.nio._

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._

import scala.collection.mutable.ArrayBuffer


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    library="jniViennaCL"
  )))
@Name(Array("viennacl::vector_base<double>"))
class VectorBase extends Pointer {

  protected val ptrs = new ArrayBuffer[Pointer]()

  override def deallocate(deallocate: Boolean): Unit = {
    super.deallocate(deallocate)
    ptrs.foreach(_.close())
  }

  // size of the vec elements
  @native @Const def size(): Int

  // size of the vec elements + padding
  @native @Const def internal_size(): Int

  // handle to the vec element buffer
  @native @Const @ByRef def handle: MemHandle

//  // add this operator in for tests many more can be added
//  @Name(Array("operator* "))
//  @native def *(i: Int): VectorMultExpression


}


