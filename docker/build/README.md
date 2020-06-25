The Spirit of this Container is so we can all build on the "same machine" e.g. no "well it worked on my machine"

Build it with 

```bash
docker build -t mahout-builder-base .
```
OR pull a convienience

```bash
docker pull rawkintrevo/mahout-builder-base
```
Get into it with

```bash
docker run -it mahout-builder-base bash
# or if you pulled convienience...
docker run -it rawkintrevo/mahout-builder-base bash 
```

Save your commands in the Dockerfile

Then maybe do something like

```bash
cd mahout 
git checkout build-cleanup
git pull # if you pulled convienience, its likely out of date.
mvn clean package -DskipTests
```

The end goal is to be able to build / post RCs from this.
