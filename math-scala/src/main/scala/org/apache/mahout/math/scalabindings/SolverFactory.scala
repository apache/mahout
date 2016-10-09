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
package org.apache.mahout.math.scalabindings

import java.io.File

import org.apache.mahout.logging._

import scala.reflect.ClassTag
import scala.reflect.runtime._
import scala.reflect._


class SolverFactory {

  private final implicit val log = getLog(this.getClass)

  // just temp for quick POC
  val classMap: Map[String,String] =
    Map(("GPUMMul"->"org.apache.mahout.viennacl.opencl.GPUMMul"),
        ("OMPMMul"->"org.apache.mahout.viennacl.openmp.OMPMMul"))
}
object SolverFactory extends SolverFactory {

    // default is JVM
    var clazz: MMBinaryFunc = MMul

    // eventually match on implicit Classtag . for now.  just take as is.
    // this is a bit hacky, Shoud not be doing onlytry/catch here..
    def getOperator[C: ClassTag]: MMBinaryFunc = {

      try {
        println("creating org.apache.mahout.viennacl.opencl.GPUMMul solver")
        clazz = Class.forName("org.apache.mahout.viennacl.opencl.GPUMMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
        println("successfully created org.apache.mahout.viennacl.opencl.GPUMMul solver")

      } catch {
        case x: Exception =>
          println(s" Error creating class: GPUMMul attempting OpenMP version")
          println(x.getMessage)
          try {
            // attempt to instantiate the OpenMP version, assuming we’ve
            // created a separate OpenMP-only module (none exist yet)
            println("creating org.apache.mahout.viennacl.openmp.OMPMMul solver")
            clazz = Class.forName("org.apache.mahout.viennacl.openmp.OMPMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
            println("successfully created org.apache.mahout.viennacl.openmp.OMPMMul solver")

          } catch {
            case x: Exception =>
              // fall back to JVM Dont need to Dynamicly assign MMul is in the same package.
              println(s" Error creating class: OMPMMul.. returning JVM MMul")
              clazz = MMul
          }
        }
      clazz
    }
}
