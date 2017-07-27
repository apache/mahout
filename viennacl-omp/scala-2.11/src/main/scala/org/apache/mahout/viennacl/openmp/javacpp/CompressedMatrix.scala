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
package org.apache.mahout.viennacl.openmp.javacpp

import java.nio._

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._

import scala.collection.mutable.ArrayBuffer


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    include = Array("compressed_matrix.hpp"),
    library="jniViennaCL"
  )))
@Name(Array("viennacl::compressed_matrix<double>"))
final class CompressedMatrix(defaultCtr: Boolean = true) extends Pointer {

  protected val ptrs = new ArrayBuffer[Pointer]()

  // call this after set or better TODO: yet wrap set() in a public method that will call this
  def registerPointersForDeallocation(p:Pointer): Unit = {
    ptrs += p
  }

  override def deallocate(deallocate: Boolean): Unit = {
    super.deallocate(deallocate)
     ptrs.foreach(_.close())
  }

  if (defaultCtr) allocate()

  def this(nrow: Int, ncol: Int) {
    this(false)
    allocate(nrow, ncol, new Context)
  }

  def this(nrow: Int, ncol: Int, ctx: Context) {
    this(false)
    allocate(nrow, ncol, ctx)
  }

  def this(nrow: Int, ncol: Int, nonzeros: Int) {
    this(false)
    allocate(nrow, ncol, nonzeros, new Context)
  }

  def this(nrow: Int, ncol: Int, nonzeros: Int, ctx: Context) {
    this(false)
    allocate(nrow, ncol, nonzeros, ctx)
  }

  def this(pe: ProdExpression) {
    this(false)
    allocate(pe)
  }

  @native protected def allocate()

  @native protected def allocate(nrow: Int, ncol: Int, nonzeros: Int, @ByVal ctx: Context)

  @native protected def allocate(nrow: Int, ncol: Int, @ByVal ctx: Context)

  @native protected def allocate(@Const @ByRef pe: ProdExpression)

//  @native protected def allocate(db: DoubleBuffer)
//
//  @native protected def allocate(ib: IntBuffer)

  // Warning: apparently there are differences in bit interpretation between OpenCL and everything
  // else for unsigned int type. So, for OpenCL backend, rowJumper and colIndices have to be packed
  // with reference to that cl_uint type that Vienna-CL defines.
  @native def set(@Cast(Array("const void*")) rowJumper: IntBuffer,
                  @Cast(Array("const void*")) colIndices: IntBuffer,
                  @Const elements: DoubleBuffer,
                  nrow: Int,
                  ncol: Int,
                  nonzeros: Int
                 )

  /** With javacpp pointers. */
  @native def set(@Cast(Array("const void*")) rowJumper: IntPointer,
                  @Cast(Array("const void*")) colIndices: IntPointer,
                  @Const elements: DoublePointer,
                  nrow: Int,
                  ncol: Int,
                  nonzeros: Int
                 )

  @Name(Array("operator="))
  @native def :=(@Const @ByRef pe: ProdExpression)

  @native def generate_row_block_information()

  /** getters for the compressed_matrix size */
  //const vcl_size_t & size1() const { return rows_; }
  @native def size1: Int
  //const vcl_size_t & size2() const { return cols_; }
  @native def size2: Int
  //const vcl_size_t & nnz() const { return nonzeros_; }
  @native def nnz: Int
  //const vcl_size_t & blocks1() const { return row_block_num_; }
 // @native def blocks1: Int

  /** getters for the compressed_matrix buffers */
  //const handle_type & handle1() const { return row_buffer_; }
  @native @Const @ByRef def handle1: MemHandle
  //const handle_type & handle2() const { return col_buffer_; }
  @native @Const @ByRef def handle2: MemHandle
  //const handle_type & handle3() const { return row_blocks_; }
  @native @Const @ByRef def handle3: MemHandle
  //const handle_type & handle() const { return elements_; }
  @native @Const @ByRef def handle: MemHandle

}

object CompressedMatrix {
  Context.loadLib()
}
