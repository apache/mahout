package org.apache.mahout.math.scalabindings

import org.apache.mahout.logging._

import scala.reflect.ClassTag
import scala.reflect.runtime._
import scala.reflect._

/**
  * Created by andrew on 9/1/16.
  */
class SolverFactory {

  private final implicit val log = getLog(this.getClass)

  // just temp for quick POC
  val classMap: Map[String,String] =
    Map(("GPUMMul"->"org.apache.mahout.viennacl.vcl.GPUMMul"))

}
object SolverFactory extends SolverFactory {

    var clazz: MMBinaryFunc = _

    // eventually match on implicit Classtag . for now.  just take as is.
    def getOperator[C: ClassTag]: MMBinaryFunc = {

      try {
        // scala equivalent of the following
        error("creating: "+classMap("GPUMMul"))

        val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader)
        val module = runtimeMirror.staticModule("org.apache.mahout.viennacl.vcl.GPUMMul")
        val clazz = runtimeMirror.reflectModule(module)

//        clazz = Class.forName(classMap("GPUMMul")).newInstance().asInstanceOf[MMBinaryFunc]
        error("successfully created org.apache.mahout.viennacl.vcl.GPUMMul solver")

      } catch {
        case x:Throwable =>
          error(s" Error creating class: $classMap(GPUMMul) attempting OpenMP version")
          x.printStackTrace()
        try {
          // attempt to instantiate the OpenMP version, assuming we’ve
          // created a separate OpenMP-only module
         val clazz = Class.forName(classMap("OMPMMul")).newInstance().asInstanceOf[MMBinaryFunc]
          error("successfully created org.apache.mahout.viennacl.ocl.OMPMMul solver")

        } catch {
          case _:Throwable =>
            error(s" Error creating class: $classMap(OMPMMul).. Exiting")
            System.exit(1)
         val clazz = org.apache.mahout.math.scalabindings.MMul
        }
      }
      clazz
    }
}
