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
package org.apache.mahout.viennacl.opencl.javacpp

import org.bytedeco.javacpp.{Loader, Pointer}
import org.bytedeco.javacpp.annotation._

/**
  * This assumes viennacl 1.7.1 is installed, which in ubuntu Xenial defaults to
  * /usr/include/viennacl, and is installed via
  * {{{
  *   sudo apt-get install libviennacl-dev
  * }}}
  *
  * @param mtype
  */
@Properties(Array(
  new Platform(
    includepath = Array("/usr/include/viennacl"),
    include = Array("matrix.hpp", "compressed_matrix.hpp"),
    define = Array("VIENNACL_WITH_OPENCL", "VIENNACL_WITH_OPENMP"),
    compiler = Array("fastfpu","viennacl"),
    link = Array("OpenCL"),
    library = "jniViennaCL"
  )))
@Namespace("viennacl")
@Name(Array("context"))
final class Context(mtype: Int = Context.MEMORY_NOT_INITIALIZED) extends Pointer {

  import Context._

  if (mtype == MEMORY_NOT_INITIALIZED)
    allocate()
  else
    allocate(mtype)

  @native protected def allocate()

  @native protected def allocate(@Cast(Array("viennacl::memory_types")) mtype: Int)

  @Name(Array("memory_type"))
  @Cast(Array("int"))
  @native def memoryType: Int

}

object Context {

  def loadLib() = Loader.load(classOf[Context])

  loadLib()

  /* Memory types. Ported from VCL header files. */
  val MEMORY_NOT_INITIALIZED = 0
  val MAIN_MEMORY = 1
  val OPENCL_MEMORY = 2
  val CUDA_MEMORY = 3

}
