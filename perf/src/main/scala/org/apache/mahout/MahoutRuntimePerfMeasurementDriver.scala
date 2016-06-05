import scala.concurrent.duration._
import akka.actor.{Actor, ActorLogging, ActorSystem, Props}
import akka.io.IO
import akka.pattern.ask
import akka.util.Timeout
import org.json4s._
import org.json4s.native.JsonMethods._
import org.json4s.native.Serialization.{read, write}
import spray.can.Http
import spray.httpx.Json4sSupport
import spray.routing._
import spray.can.server.Stats
import spray.http.StatusCodes._
import basic._
import com.mlh.spraysample.{basic, versioning}
import _root_.versioning._
import com.mlh.spraysample.basic.SpraySampleService

object MahoutRuntimePerfMeasurementDriver extends App {
  implicit val system = ActorSystem("spray-run-time-performance-measurement-system")

  /* Use Akka to create our Spray Service */
  val service = system.actorOf(Props[SprayRunTimePerformanceMeasurementActor], "spray-tun-time-performance-measurement-service")

  /* and bind to Akka's I/O interface */
  IO(Http) ! Http.Bind(service, system.settings.config.getString("app.interface"), system.settings.config.getInt("app.port"))

}

/* Our Server Actor is pretty lightweight; simply mixing in our route trait and logging */
class SprayRunTimePerformanceMeasurementActor extends Actor with SpraySampleService with ActorLogging {
  def actorRefFactory = context
  def receive = runRoute(spraysampleRoute)
}
