/*
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

package org.apache.mahout.math.scalabindings

import org.apache.mahout.math.{Vector, Matrix}

/**
 * R-like operators. Declare <code>import RLikeOps._</code> to enable.
 */
object RLikeOps {

  implicit def double2Scalar(x:Double) = new RLikeDoubleScalarOps(x)

  implicit def v2vOps(v: Vector) = new RLikeVectorOps(v)

  implicit def el2elOps(el: Vector.Element) = new ElementOps(el)

  implicit def el2Double(el:Vector.Element) = el.get()

  implicit def m2mOps(m: Matrix) = new RLikeMatrixOps(m)


}
