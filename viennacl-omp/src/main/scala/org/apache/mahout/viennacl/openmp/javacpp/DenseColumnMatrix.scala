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

import org.bytedeco.javacpp.DoublePointer
import org.bytedeco.javacpp.annotation._

/**
  * ViennaCL dense matrix, column-major. This is an exact duplication of [[DenseRowMatrix]], and
  * is only different in the materialized C++ template name. Unfortunately I so far have not figured
  * out how to handle it with.
  *
  * Also, the [[Platform.library]] does not get inherited for some reason, and we really want to
  * collect all class mappings in the same one libjni.so, so we have to repeat this `library` defi-
  * nition in every mapped class in this package. (One .so per package convention).
  */
@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform (
    include=Array("matrix.hpp"),
    library="jniViennaCL"
  )))
@Name(Array("viennacl::matrix<double,viennacl::column_major>"))
final class DenseColumnMatrix(initDefault:Boolean = true) extends MatrixBase {

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

}

object DenseColumnMatrix {
  Context.loadLib()
}