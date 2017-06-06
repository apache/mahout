package org.apache.mahout.h2o.common

import scala.reflect.ClassTag
import org.apache.hadoop.io._
import java.util.Arrays

class DrmMetadata(

    /** Writable  key type as a sub-type of Writable */
    val keyTypeWritable: Class[_],

    /** Value writable type, as a sub-type of Writable */
    val valueTypeWritable: Class[_]

    ) {

  import DrmMetadata._

  val (

      /** Actual drm key class tag once converted out of writable */
      keyClassTag: ClassTag[_],

      /** Conversion from Writable to value type of the DRM key */
      keyW2ValFunc: ((Writable) => Any)

      ) = keyTypeWritable match {
    case cz if cz == classOf[IntWritable] => ClassTag.Int -> w2int _
    case cz if cz == classOf[LongWritable] => ClassTag.Long -> w2long _
    case cz if cz == classOf[DoubleWritable] => ClassTag.Double -> w2double _
    case cz if cz == classOf[FloatWritable] => ClassTag.Float -> w2float _
    case cz if cz == classOf[Text] => ClassTag(classOf[String]) -> w2string _
    case cz if cz == classOf[BooleanWritable] => ClassTag(classOf[Boolean]) -> w2bool _
    case cz if cz == classOf[BytesWritable] => ClassTag(classOf[Array[Byte]]) -> w2bytes _
    case _ => throw new IllegalArgumentException(s"Unsupported DRM key type:${keyTypeWritable.getName}")
  }

}

object DrmMetadata {

  private[common] def w2int(w: Writable) = w.asInstanceOf[IntWritable].get()

  private[common] def w2long(w: Writable) = w.asInstanceOf[LongWritable].get()

  private[common] def w2double(w: Writable) = w.asInstanceOf[DoubleWritable].get()

  private[common] def w2float(w: Writable) = w.asInstanceOf[FloatWritable].get()

  private[common] def w2string(w: Writable) = w.asInstanceOf[Text].toString()

  private[common] def w2bool(w: Writable) = w.asInstanceOf[BooleanWritable].get()

  private[common] def w2bytes(w: Writable) = Arrays.copyOf(w.asInstanceOf[BytesWritable].getBytes(),
                                                           w.asInstanceOf[BytesWritable].getLength())
}
