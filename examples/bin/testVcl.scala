import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf



object TestVCL {

  def main(args: Array[String]) {
    run()
  }

  private def run() {
    val sparkConf = new SparkConf()
    sparkConf.setAppName("TestJob")
    sparkConf.set("spark.cores.max", "8")
    sparkConf.set("spark.storage.memoryFraction", "0.1")
    sparkConf.set("spark.shuffle.memoryFracton", "0.2")
    sparkConf.set("spark.executor.memory", "2g")
    sparkConf.setJars(List("target/scala-2.10/spark-test-assembly-1.0.jar"))
    sparkConf.setMaster(s"spark://dev1.dev.pulse.io:7077")
    sparkConf.setSparkHome("/home/pulseio/spark/current")
    val sc = new SparkContext(sparkConf)

    sc.stop()
  }

}


