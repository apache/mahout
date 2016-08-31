package org.apache.mahout.viennacl.omp.javacpp

import org.bytedeco.javacpp.Pointer
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
