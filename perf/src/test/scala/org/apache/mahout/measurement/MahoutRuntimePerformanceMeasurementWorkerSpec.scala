package org.apache.mahout.measurement

import org.scalatest.{BeforeAndAfterAll, FreeSpecLike, Matchers}

/**
  * Created by saikat on 6/5/16.
  */
class MahoutRuntimePerformanceMeasurementWorkerSpec extends TestKit(ActorSystem("MahoutRuntimePerformanceMeasurementWorkerSpec"))
  with ImplicitSender
  with FreeSpecLike
  with Matchers
  with BeforeAndAfterAll {
  import org.apache.mahout.measurement.service.MahoutRuntimePerformanceActorgit ._

  val actorRef = TestActorRef[MahoutRuntimePerformanceMeasurementActor]

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

