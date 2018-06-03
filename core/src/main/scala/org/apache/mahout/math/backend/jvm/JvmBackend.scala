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
package org.apache.mahout.math.backend.jvm

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.math.backend.Backend
import org.apache.mahout.math.scalabindings.MMul

import scala.collection.Map
import scala.reflect._

object JvmBackend extends Backend {

  import org.apache.mahout.math.backend.incore._

  /**
    * If backend has loaded (lazily) ok and verified its availability/functionality,
    * this must return `true`.
    *
    * @return `true`
    */
  override def isAvailable: Boolean = true

  // TODO: In a future release, Refactor MMul optimizations into this object
  override protected[backend] val solverMap: Map[ClassTag[_], Any] = Map(
    classTag[MMulSolver] → MMul
    //    classTag[MMulDenseSolver] → MMul,
    //    classTag[MMulSparseSolver] → MMul,
    //    classTag[AtASolver] → new AtASolver {
    //      override def apply(a: Matrix, r: Option[Matrix]): Matrix = MMul(a.t, a, r)
    //    }// ,
    //    classTag[AtADenseSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a.t, a, r) },
    //    classTag[AtASparseSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a.t, a, r) },
    //    classTag[AAtSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a, a.t, r) },
    //    classTag[AAtDenseSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a, a.t, r) },
    //    classTag[AAtSparseSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a, a.t, r) }
  )
  validateMap()

  private val mmulSolver = new MMulSolver with MMulDenseSolver with MMulSparseSolver {
    override def apply(a: Matrix, b: Matrix, r: Option[Matrix]): Matrix = MMul(a, b, r)
  }

  private val ataSolver = new AtASolver with AtADenseSolver with AtASparseSolver {
    override def apply(a: Matrix, r: Option[Matrix]): Matrix = MMul(a.t, a, r)
  }

  private val aatSolver = new AAtSolver {
    override def apply(a: Matrix, r: Option[Matrix]): Matrix = MMul(a, a.t, r)
  }
}
