package com.mlh.spraysample
package basic

import org.scalatest.FreeSpecLike
import org.scalatest.Matchers
import org.scalatest.BeforeAndAfterAll

import akka.actor.ActorSystem
import akka.testkit.TestActorRef
import akka.testkit.TestKit
import akka.testkit.ImplicitSender

import scala.concurrent.duration._
import scala.concurrent.Await
import akka.pattern.ask

class WorkerSpec extends TestKit(ActorSystem("WorkerSpec"))
    with ImplicitSender
    with FreeSpecLike
    with Matchers
    with BeforeAndAfterAll {
  import WorkerActor._

  val actorRef = TestActorRef[WorkerActor]

  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  "Worker" - {
    "creates" - {
      "receive Ok" - {
        val future = actorRef ! Create(new Foo("bar"))
        expectMsgClass(classOf[Ok])
      }
    }
  }
}
