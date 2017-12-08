package org.apache.mahout.cylon-example.localengine

import java.awt.image.BufferedImage
import java.util.Properties

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}

trait AbstractKafkaLocalEngine extends AbstractLocalEngine{

  var kafkaProps: Properties = new Properties()
  kafkaProps.put("bootstrap.servers", "localhost:9092")
  kafkaProps.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
  kafkaProps.put("value.serializer", "org.apache.kafka.common.serialization.ByteArraySerializer")

  var producer: KafkaProducer[String, Array[Byte]]= _

  def writeToKafka(topic: String, key: String, data: Array[Byte]): Unit = {
    val record = new ProducerRecord(topic, key, data)
    producer.send(record)
  }

  def setupKafkaProducer(): Unit = {
    producer = new KafkaProducer[String, Array[Byte]](kafkaProps)
    logger.info(s"Kafka Producer Established on Boot Strap Server ${kafkaProps.getProperty("bootstrap.servers")}")
  }

}
