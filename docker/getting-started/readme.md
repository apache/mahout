# Getting Started With Mahout Using Apache Zeppelin

## Setting Environment Variables

```
export CONTAINER_NAME=$USER/mahoutgui
export VERSION=14.3-SNAPSHOT
```

## Building Container

```bash
docker build -t $CONTAINER_NAME:$VERSION .
```

## Running Container

```bash
docker run -p 8080:8080 --rm --name whatever $CONTAINER_NAME:$VERSION
```

Then surf to http://localhost:8080
