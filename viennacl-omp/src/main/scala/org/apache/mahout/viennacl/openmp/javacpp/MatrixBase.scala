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

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation._

import scala.collection.mutable.ArrayBuffer


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
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
