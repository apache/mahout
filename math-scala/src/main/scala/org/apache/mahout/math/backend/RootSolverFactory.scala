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
package org.apache.mahout.math.backend

import org.apache.mahout.logging._
import org.apache.mahout.math.backend.jvm.JvmBackend
import org.apache.mahout.math.scalabindings.{MMul, _}

import scala.collection._
import scala.reflect.{ClassTag, classTag}


final object RootSolverFactory extends SolverFactory {

  import org.apache.mahout.math.backend.incore._

  implicit val logger = getLog(RootSolverFactory.getClass)

  private val solverTagsToScan =
    classTag[MMulSolver] ::
      classTag[MMulSparseSolver] ::
      classTag[MMulDenseSolver] ::
      Nil

  private val defaultBackendPriority =
    JvmBackend.getClass.getName :: Nil

  private def initBackends(): Unit = {

  }

  ////////////////////////////////////////////////////////////

  // TODO: MAHOUT-1909: lazy initialze the map. Query backends. Build resolution rules.
  override protected[backend] val solverMap = new mutable.HashMap[ClassTag[_], Any]()
  validateMap()


  // default is JVM
  var clazz: MMBinaryFunc = MMul

  // eventually match on implicit Classtag . for now.  just take as is.
  // this is a bit hacky, Shoud not be doing onlytry/catch here..
  def getOperator[C: ClassTag]: MMBinaryFunc = {

    try {
      // TODO: fix logging properties so that we're not mimicing as we are here.
      println("[INFO] Creating org.apache.mahout.viennacl.opencl.GPUMMul solver")
      clazz = Class.forName("org.apache.mahout.viennacl.opencl.GPUMMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
      println("[INFO] Successfully created org.apache.mahout.viennacl.opencl.GPUMMul solver")

    } catch {
      case x: Exception =>
        println("[WARN] Unable to create class GPUMMul: attempting OpenMP version")
        // println(x.getMessage)
        try {
          // attempt to instantiate the OpenMP version, assuming we’ve
          // created a separate OpenMP-only module (none exist yet)
          println("[INFO] Creating org.apache.mahout.viennacl.openmp.OMPMMul solver")
          clazz = Class.forName("org.apache.mahout.viennacl.openmp.OMPMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
          println("[INFO] Successfully created org.apache.mahout.viennacl.openmp.OMPMMul solver")

        } catch {
          case x: Exception =>
            // fall back to JVM Dont need to Dynamicly assign MMul is in the same package.
            println("[INFO] Unable to create class OMPMMul: falling back to java version")
            clazz = MMul
        }
    }
    clazz
  }
}
