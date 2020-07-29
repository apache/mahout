<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

The Spirit of this Container is so we can all build on the "same machine" e.g. no "well it worked on my machine"


Build it with

docker build -t mahout-builder-base .
OR pull a convienience

docker pull rawkintrevo/mahout-builder-base
Get into it with

```
docker run -it mahout-builder-base bash
# or if you pulled convienience...
docker run -it rawkintrevo/mahout-builder-base bash 
```
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

