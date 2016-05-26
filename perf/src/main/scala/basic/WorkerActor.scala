package com.mlh.spraysample
package basic

import akka.actor.{ Actor, ActorSystem, Props, ActorLogging }
import akka.io.IO
import akka.routing._
import org.json4s._
import org.json4s.native.JsonMethods._

object WorkerActor {
  case class Ok(id: Int)
  case class Create(foo: Foo)
}

class WorkerActor extends Actor with ActorLogging {
  import WorkerActor._

  def receive = {
    case Create(foo) => {
      log.info(s"Create ${foo}")
      sender ! Ok(util.Random.nextInt(10000))
    }
  }
}

