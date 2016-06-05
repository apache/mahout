package org.apache.mahout.measurement.service

import akka.actor.{Actor, ActorLogging, ActorSystem, Props}
import akka.io.IO
import akka.routing._
import com.mlh.spraysample.basic.Foo
import com.mlh.spraysample.basic.WorkerActor.{Create, Ok}
import org.json4s._
import org.json4s.native.JsonMethods._

object MahoutRuntimePerformanceActor {
  case class Ok(id: Int)
  case class Create(foo: Foo)
}

class MahoutRuntimePerformanceActor extends Actor with ActorLogging {
  import MahoutRuntimePerformanceActor._

  def receive = {
    case Create(foo) => {
      log.info(s"Create ${foo}")
      sender ! Ok(util.Random.nextInt(10000))
    }
  }
}
