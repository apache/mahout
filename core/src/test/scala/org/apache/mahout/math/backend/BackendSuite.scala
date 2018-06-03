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
package org.apache.mahout.math.backend

import org.apache.mahout.math.backend.jvm.JvmBackend
import org.scalatest.{FunSuite, Matchers}

import scala.collection.mutable
import scala.reflect.{ClassTag, classTag}

class BackendSuite extends FunSuite with Matchers {

  test("GenericBackend") {

    trait MySolverTrait1 { def myMethod1 = Unit }


    trait MySolverTrait2

    class MySolverImpl1 extends MySolverTrait1 {
    }

    class MySolverImpl2 extends MySolverTrait2

    // My dummy backend supporting to trait solvers filled with 2 dummy implementations of these
    // traits should be able to serve based on their solver traits.
    val myBackend = new Backend {

      override def isAvailable: Boolean = true

      override val solverMap = new mutable.HashMap[ClassTag[_], Any]()

      solverMap ++= Map(
        classTag[MySolverTrait1] → new MySolverImpl1,
        classTag[MySolverTrait2] → new MySolverImpl2
      )

      validateMap()
    }

    myBackend.getSolver shouldBe None

    val mySolver1 = myBackend.getSolver[MySolverTrait1]

    // This is indeed solver1 trait type:
    mySolver1.get.myMethod1
    mySolver1.get.isInstanceOf[MySolverImpl1] shouldBe true

    // Validator should not allow non-subclasses in implementation.
    an [IllegalArgumentException] mustBe thrownBy {
      myBackend.solverMap(classTag[MySolverTrait2]) = 0
      myBackend.validateMap()
    }
  }

  test("JvmBackend") {
    // Just create JVM backend and validate.
    JvmBackend.validateMap()
  }

}
