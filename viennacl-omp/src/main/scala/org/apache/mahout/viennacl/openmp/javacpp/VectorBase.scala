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


