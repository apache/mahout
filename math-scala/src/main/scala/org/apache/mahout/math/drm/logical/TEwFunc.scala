/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.drm.logical

/**
 * Trait denoting logical operators providing elementwise operations that work as unary operators
 * on each element of a matrix.
 */
trait TEwFunc {

  /** Apply to degenerate elments? */
  def evalZeros: Boolean

  /** the function itself */
  def f: (Double) => Double

  /**
   * Self assignment ok? If yes, may cause side effects if works off non-serialized cached object
   * tree!
   */
  def selfAssignOk: Boolean = false
}
