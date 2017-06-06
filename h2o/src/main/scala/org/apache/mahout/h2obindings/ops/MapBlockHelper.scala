/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings.ops

import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm.BlockMapFunc
import scala.reflect.ClassTag

import water.fvec.{Vec,NewChunk}
import water.parser.ValueString

object MapBlockHelper {
  def exec[K: ClassTag, R: ClassTag](bmf: Object, in: Matrix, startlong: Long, labels: Vec, nclabel: NewChunk): Matrix = {
    val i = implicitly[ClassTag[Int]]
    val l = implicitly[ClassTag[Long]]
    val s = implicitly[ClassTag[String]]

    val inarray = implicitly[ClassTag[K]] match {
      case `i` => val startint: Int = startlong.asInstanceOf[Int]
        startint until (startint + in.rowSize) toArray
      case `l` => startlong until (startlong + in.rowSize) toArray
      case `s` =>
        val arr = new Array[String](in.rowSize)
        val vstr = new ValueString
        for (i <- 0 until in.rowSize) {
          arr(i) = labels.atStr(vstr, i + startlong).toString
        }
        arr
    }

    val _bmf = bmf.asInstanceOf[BlockMapFunc[K,R]]
    val out = _bmf((inarray.asInstanceOf[Array[K]], in))

    implicitly[ClassTag[R]] match {
      case `s` =>
        val vstr = new ValueString
        for (str <- out._1) {
          nclabel.addStr(vstr.setTo(str.asInstanceOf[String]))
        }
      case _ =>
    }
    out._2
  }
}

