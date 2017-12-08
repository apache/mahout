package org.apache.mahout.cylon-example.frameprocessors

import java.io._

/**
  * This is an example of an extremely stupid way (and consequently the way Flink does RocksDB) to handle the JNI problem.
  *
  * DO NOT USE!!
  *
  * Basically we include libopencv_java330.so in src/main/resources so then it creates a tmp version.
  *
  * I deleted from it resources, so this would fail.  Only try it for academic purposes. E.g. to see what stupid looks like.
  *
  */
object NativeUtils {
  // heavily based on https://github.com/adamheinrich/native-utils/blob/master/src/main/java/cz/adamh/utils/NativeUtils.java
  def loadOpenCVLibFromJar() = {

    val temp = File.createTempFile("libopencv_java330", ".so")
    temp.deleteOnExit()

    val inputStream= getClass().getResourceAsStream("/libopencv_java330.so")

    import java.io.FileOutputStream
    val os = new FileOutputStream(temp)
    var readBytes: Int = 0
    var buffer = new Array[Byte](1024)
    try {
      while ({(readBytes = inputStream.read(buffer))
               readBytes != -1}) {
        os.write(buffer, 0, readBytes)
      }
    }
    finally {
      // If read/write fails, close streams safely before throwing an exception
      os.close()
      inputStream.close
    }

    System.load(temp.getAbsolutePath)
  }

}
