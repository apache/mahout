package com.mlh.spraysample

import scala.concurrent.duration._

import akka.actor.{ Actor, ActorLogging, ActorSystem, Props }
import akka.io.IO
import akka.pattern.ask
import akka.util.Timeout
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization.{ read, write }
import spray.can.Http
import spray.httpx.Json4sSupport
import spray.routing._
import spray.can.server.Stats
import spray.http.StatusCodes._
import basic._
import versioning._

object Boot extends App {
  implicit val system = ActorSystem("spray-sample-system")

  /* Use Akka to create our Spray Service */
  val service = system.actorOf(Props[SpraySampleActor], "spray-sample-service")

  /* and bind to Akka's I/O interface */
  IO(Http) ! Http.Bind(service, system.settings.config.getString("app.interface"), system.settings.config.getInt("app.port"))

}

/* Our Server Actor is pretty lightweight; simply mixing in our route trait and logging */
class SpraySampleActor extends Actor with SpraySampleService with ActorLogging {
  def actorRefFactory = context
  def receive = runRoute(spraysampleRoute)
}
