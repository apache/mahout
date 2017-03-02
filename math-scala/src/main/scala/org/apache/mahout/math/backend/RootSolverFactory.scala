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
import org.apache.mahout.math.scalabindings.{MMBinaryFunc, MMul, _}

import scala.collection._
import scala.reflect.{ClassTag, classTag}


final object RootSolverFactory extends SolverFactory {

  import org.apache.mahout.math.backend.incore._

  private implicit val logger = getLog(RootSolverFactory.getClass)

  private val solverTagsToScan =
    classTag[MMulSolver] ::
      classTag[MMulSparseSolver] ::
      classTag[MMulDenseSolver] ::
      Nil

  private val defaultBackendPriority =
    JvmBackend.getClass.getName :: Nil

  private def initBackends(): Unit = {

  }

  // TODO: MAHOUT-1909: Cache Modular Backend solvers after probing
  // That is, lazily initialize the map, query backends, and build resolution rules.
  override protected[backend] val solverMap = new mutable.HashMap[ClassTag[_], Any]()

  validateMap()

  // Default solver is JVM
  var clazz: MMBinaryFunc = MMul

  // TODO: Match on implicit Classtag

  def getOperator[C: ClassTag]: MMBinaryFunc = {

    try {
      logger.info("Creating org.apache.mahout.viennacl.opencl.GPUMMul solver")
      clazz = Class.forName("org.apache.mahout.viennacl.opencl.GPUMMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
      logger.info("Successfully created org.apache.mahout.viennacl.opencl.GPUMMul solver")

    } catch {
      case x: Exception =>
        logger.info("Unable to create class GPUMMul: attempting OpenMP version")
        try {
          // Attempt to instantiate the OpenMP version, assuming we’ve
          // created a separate OpenMP-only module (none exist yet)
          logger.info("Creating org.apache.mahout.viennacl.openmp.OMPMMul solver")
          clazz = Class.forName("org.apache.mahout.viennacl.openmp.OMPMMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
          logger.info("Successfully created org.apache.mahout.viennacl.openmp.OMPMMul solver")

        } catch {
          case xx: Exception =>
            logger.info(xx.getMessage)
            // Fall back to JVM; don't need to dynamically assign since MMul is in the same package.
            logger.info("Unable to create class OMPMMul: falling back to java version")
            clazz = MMul
        }
    }
    clazz
  }
}
