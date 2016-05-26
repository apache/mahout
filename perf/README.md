# spray-sample #

An example api app that leverages [Spray](spray.io) and [Akka](akka.io). You may also be interested
in [stockman](https://github.com/mhamrah/stockman) which provides a richer example application with more functionality.

## Running
Clone to your computer and run via sbt:

```
sbt run
```

## About
The app simply exposes some basic endpoints.  A POST request to /entity with a Content-Type header of application/json will use Spray's json4s marhsalling support to create a JObject which gets passed to another Actor, with the HttpResponse returned to Spray via a future.

Here are some curl commands you can try:

```
curl -v http://localhost:8080/entity
curl -v http://localhost:8080/entity/1234
curl -v -X POST http://localhost:8080/entity -H "Content-Type: application/json" -d "{ \"property\" : \"value\" }"
```

## Spray giter8 Template

If you'd like to get started with spray, you can bootstrap your own spray project with the spray branch of my sbt [giter8](https://github.com/n8han/giter8) project:

```
$ g8 mhamrah/sbt -b spray
```
