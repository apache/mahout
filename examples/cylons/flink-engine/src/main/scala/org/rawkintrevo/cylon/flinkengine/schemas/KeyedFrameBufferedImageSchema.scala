package org.apache.mahout.cylon-example.flinkengine.schemas

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import javax.imageio.ImageIO

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.streaming.api.scala.createTypeInformation
import org.apache.flink.streaming.util.serialization.{KeyedDeserializationSchema, KeyedSerializationSchema}
import org.slf4j.{Logger, LoggerFactory}


class KeyedFrameBufferedImageSchema
  extends KeyedDeserializationSchema[((String, Int), BufferedImage)]
      with KeyedSerializationSchema[((String, Int), BufferedImage)] {

    val logger: Logger = LoggerFactory.getLogger(classOf[KeyedBufferedImageSchema])

    def isEndOfStream(nextElement: ((String, Int), BufferedImage)): Boolean = {
      false
    }

    def getProducedType: TypeInformation[((String, Int), BufferedImage)] = createTypeInformation[((String, Int), BufferedImage)]

    def deserialize(messageKey: Array[Byte],
                    message: Array[Byte],
                    topic: String,
                    partition: Int,
                    offset: Long): ((String, Int), BufferedImage) = {
      val img = ImageIO.read(new ByteArrayInputStream(message))
      val keyStrArray: Array[String] = new String(messageKey, "UTF-8").split("-")
      val key = keyStrArray(0)
      val frame = keyStrArray(1).toInt
      ((key, frame), img)
    }

    override def serializeKey(t: ((String, Int), BufferedImage)): Array[Byte] =  {
      s"${t._1._1}-${t._1._2}-raw_image".getBytes("UTF-8")
    }

    override def serializeValue(t: ((String, Int), BufferedImage)): Array[Byte] =  {
      val baos = new ByteArrayOutputStream()
      ImageIO.write(t._2, "jpg", baos)
      baos.toByteArray
    }

    override def getTargetTopic(t: ((String, Int), BufferedImage)): String = {
      /** Says in the code comments this is supposed to be optional:
        * https://github.com/apache/flink/blob/master/flink-connectors/flink-connector-kafka-base/src/main/java/org/apache/flink/streaming/util/serialization/KeyedSerializationSchema.java#L49
        *
        * Found here that we're never using it anyway so return null
        * https://github.com/apache/flink/blob/master/flink-connectors/flink-connector-kafka-base/src/main/java/org/apache/flink/streaming/util/serialization/TypeInformationKeyValueSerializationSchema.java#L179
        */
      null
    }

  }