package org.apache.mahout.viennacl.vcl.javacpp

import org.bytedeco.javacpp.{Loader, Pointer}
import org.bytedeco.javacpp.annotation._


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    library = "jniViennaCL")
  ))
@Namespace("viennacl::backend")
@Name(Array("mem_handle"))
class MemHandle extends Pointer {

  allocate()

  @native def allocate()
}

object MemHandle {

  def loadLib() = Loader.load(classOf[MemHandle])

  loadLib()

  /* Memory types. Ported from VCL header files. */
  val MEMORY_NOT_INITIALIZED = 0
  val MAIN_MEMORY = 1
  val OPENCL_MEMORY = 2
  val CUDA_MEMORY = 3

}
