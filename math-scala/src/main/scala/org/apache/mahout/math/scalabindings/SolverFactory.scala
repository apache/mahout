package org.apache.mahout.math.scalabindings

import java.io.File

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
        info("creating: "+classMap("GPUMMul"))

        // returning $MAHOUT_HOME/null/
        // val mahoutHome = System.getProperty("mahout.home")
//        val mahoutHome = "/home/andrew/sandbox/mahout"
//        var classLoader = new java.net.URLClassLoader(
//            Array(new File( mahoutHome + "/viennacl/target/" +
//              "mahout-native-viennacl_2.10-0.12.3-SNAPSHOT.jar").toURI.toURL),
//              this.getClass.getClassLoader)
//
////        System.out.println("\n\n\n")
////        (classLoader.getURLs).foreach{x => println(x).toString}
////        System.out.println("\n\n\n")
//
//        val clazzMod = classLoader.loadClass("org.apache.mahout.viennacl.vcl.GPUMMul$")

//        val clazz = clazzMod.getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]

//        println("class: "+clazz.getClass.getSimpleName+"\n\n\n")


        val clazz = Class.forName("org.apache.mahout.viennacl.vcl.GPUMMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
       // info("successfully created org.apache.mahout.viennacl.vcl.GPUMMul solver")

      } catch {
        case x: Exception =>
          error(s" Error creating class: $classMap(GPUMMul) attempting OpenMP version")
         // x.printStackTrace()
          try {
            // attempt to instantiate the OpenMP version, assuming we’ve
            // created a separate OpenMP-only module (none exist yet)
            val clazz = Class.forName("org.apache.mahout.viennacl.vcl.OMPMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
          //  info("successfully created org.apache.mahout.viennacl.ocl.OMPMMul solver")

          } catch {
            case x:Exception =>
              // fall back to JVM Dont need to Dynamically assign MMul is in the same package.
            //  error(s" Error creating class: $classMap(OMPMMul).. returning JVM MMul")
//              val clazz = Class.forName("org.apache.mahout.math.scalabindings.MMul$").getField("MODULE$").get(null).asInstanceOf[MMBinaryFunc]
//              x.printStackTrace()
              clazz = MMul
//             x.printStackTrace()
          }
        }
      clazz
    }
}
