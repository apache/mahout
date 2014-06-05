package org.apache.mahout.math.scalaframes


/**
 *
 * @author dmitriy
 */
object UnsafeUtil {
  lazy val arrayOffset = unsafe.arrayBaseOffset(classOf[Array[Byte]])
  lazy val unsafe = concurrent.util.Unsafe.instance

  //  lazy val unsafe: Unsafe = {
  //
  //    if (this.getClass.getClassLoader() == null)
  //      Unsafe.getUnsafe();
  //
  //    try {
  //      val fld: Field = classOf[Unsafe].getDeclaredField("theUnsafe");
  //      fld.setAccessible(true);
  //      fld.get(classOf[Unsafe]).asInstanceOf[Unsafe];
  //    } catch {
  //      case e: Throwable => throw new RuntimeException("no sun.misc.Unsafe", e);
  //    }
  //  }

  def setUnsafeDouble(arr: Array[Byte], x: Double, offset: Long): this.type = {
    unsafe.putDouble(arr, arrayOffset + offset, x)
    this
  }

  def getUnsafeDouble(arr: Array[Byte], offset: Long): Double = {
    unsafe.getDouble(arr, arrayOffset + offset)
  }

  def setUnsafeLong(arr: Array[Byte], x: Long, offset: Long): this.type = {
    unsafe.putLong(arr, arrayOffset + offset, x)
    this
  }

  def getUnsafeLong(arr: Array[Byte], offset: Long): Long = {
    unsafe.getLong(arr, arrayOffset + offset)
  }


}
