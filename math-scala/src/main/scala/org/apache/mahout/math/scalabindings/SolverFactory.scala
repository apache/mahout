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
    Map(("GPUMMul"->"org.apache.mahout.viennacl.vcl.GPUMMul"),
        ("OMPMMul"->"org.apache.mahout.viennacl.omp.OMPMMul"))
}
object SolverFactory extends SolverFactory {

    // default is JVM
    var clazz: MMBinaryFunc = MMul

    // eventually match on implicit Classtag . for now.  just take as is.
    def getOperator[C: ClassTag]: MMBinaryFunc = {

      try {
        println("creating org.apache.mahout.viennacl.vcl.GPUMMul solver")
        val clazz = Class.forName("org.apache.mahout.viennacl.vcl.GPUMMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
        println("successfully created org.apache.mahout.viennacl.vcl.GPUMMul solver")

      } catch {
        case x: Exception =>
          println(s" Error creating class: GPUMMul attempting OpenMP version")
          println(x.getMessage)
          try {
            // attempt to instantiate the OpenMP version, assuming we’ve
            // created a separate OpenMP-only module (none exist yet)
            println("creating org.apache.mahout.viennacl.vcl.OMPMMul solver")
            val clazz = Class.forName("org.apache.mahout.viennacl.vcl.OMPMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
            println("successfully created org.apache.mahout.viennacl.ocl.OMPMMul solver")

          } catch {
            case x: Exception =>
              // fall back to JVM Dont need to Dynamicly assign MMul is in the same package.
              println(s" Error creating class: OMPMMul.. returning JVM MMul")
              clazz = org.apache.mahout.math.scalabindings.MMul
          }
        }
      clazz
    }
}
