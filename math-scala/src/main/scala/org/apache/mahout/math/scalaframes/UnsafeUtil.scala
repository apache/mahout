package org.apache.mahout.math.scalaframes

import sun.misc.Unsafe
import java.lang.reflect.Field

/**
 *
 * @author dmitriy
 */
object UnsafeUtil {
  lazy val unsafe:Unsafe = {

    if (this.getClass.getClassLoader() == null )
      Unsafe.getUnsafe();

    try {
      val fld:Field = classOf[Unsafe].getDeclaredField("theUnsafe");
      fld.setAccessible(true);
      fld.get(classOf[Unsafe]).asInstanceOf[Unsafe];
    } catch {
      case e:Throwable => throw new RuntimeException("no sun.misc.Unsafe", e);
    }
  }

}
