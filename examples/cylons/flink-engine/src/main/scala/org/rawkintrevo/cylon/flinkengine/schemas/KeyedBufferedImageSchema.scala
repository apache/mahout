package org.apache.mahout.cylon-example.flinkengine.schemas

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import javax.imageio.ImageIO

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.streaming.api.scala.createTypeInformation
import org.apache.flink.streaming.util.serialization.{KeyedDeserializationSchema, KeyedSerializationSchema}
import org.slf4j.{Logger, LoggerFactory}


class KeyedBufferedImageSchema
  extends KeyedDeserializationSchema[(String, BufferedImage)]
    with KeyedSerializationSchema[(String, BufferedImage)] {

  val logger: Logger = LoggerFactory.getLogger(classOf[KeyedBufferedImageSchema])

  def isEndOfStream(nextElement: (String, BufferedImage)): Boolean = {
    false
  }

  def getProducedType: TypeInformation[(String, BufferedImage)] = createTypeInformation[(String, BufferedImage)]

  def deserialize(messageKey: Array[Byte],
                  message: Array[Byte],
                  topic: String,
                  partition: Int,
                  offset: Long): (String, BufferedImage) = {
    val img = ImageIO.read(new ByteArrayInputStream(message))
    (new String(messageKey, "UTF-8"), img)
  }

  override def serializeKey(t: (String, BufferedImage)): Array[Byte] =  {
    t._1.getBytes("UTF-8")
  }

  override def serializeValue(t: (String, BufferedImage)): Array[Byte] =  {
    val baos = new ByteArrayOutputStream()
    ImageIO.write(t._2, "png", baos)
    baos.flush()
    val out = baos.toByteArray
    baos.close()
    out
  }

  override def getTargetTopic(t: (String, BufferedImage)): String = {
    /** Says in the code comments this is supposed to be optional:
      * https://github.com/apache/flink/blob/master/flink-connectors/flink-connector-kafka-base/src/main/java/org/apache/flink/streaming/util/serialization/KeyedSerializationSchema.java#L49
      *
      * Found here that we're never using it anyway so return null
      * https://github.com/apache/flink/blob/master/flink-connectors/flink-connector-kafka-base/src/main/java/org/apache/flink/streaming/util/serialization/TypeInformationKeyValueSerializationSchema.java#L179
      */
    null
  }

}