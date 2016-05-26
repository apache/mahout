package com.mlh.spraysample
package basic

import org.scalatest._
import spray.testkit.ScalatestRouteTest
import spray.http.HttpEntity
import spray.http.ContentTypes
import spray.can.server.Stats
import spray.http.StatusCodes._
import org.json4s._

class MainSpec extends FreeSpec with Matchers with ScalatestRouteTest with SpraySampleService {
  def actorRefFactory = system

  "The spraysample Route" - {
    "when listing entities" - {
      "returns a JSON list" in {
        //Mix in Json4s, but only for this test
        import Json4sProtocol._

        Get("/entity") ~> spraysampleRoute ~> check {
          assert(contentType.mediaType.isApplication)

          //Check content type
          contentType.toString should include("application/json")
          //Try serializaing as a List of Foo
          val response = responseAs[List[Foo]]

          response.size should equal(2)
          response(0).bar should equal("foo1")

          //Check http status
          status should equal(OK)
        }
      }
    }
    "when posting an entity" - {
      "gives a JSON response" in {
        // FIXME: Should be able to send entity as JObject
        Post("/entity", HttpEntity(ContentTypes.`application/json`, """{"bar": "woot!"}""")) ~> spraysampleRoute ~> check {
          responseAs[String] should include("\"I got a response: Ok")
          status should equal(Created)
        }
      }
    }
  }
}

