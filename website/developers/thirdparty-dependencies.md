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
---
layout: page
title: Thirdparty Dependencies

    
---

# Adding Thirdparty Dependencies in Maven

If you have a dependency on a third party artifact that is not in Maven,
you should:


* Ask the project to add it if at all possible.  Most open source projects
want wider adoption, so this kind of request is often well received.
* If they won't add it, we may be able to add it to our Maven repo,
assuming it can be published at the ASF at all (no GPL code, for instance).
 Please ask on the mailing list first.
* Assuming it can be, then you need to sign and deploy the artifacts, as
described below:

*mvn gpg:sign-and-deploy-file -Durl=https://repository.apache.org/service/local/staging/deploy/maven2 -DrepositoryId=apache.releases.https -DgroupId=org.apache.mahout.foobar -DartifactId=foobar -Dversion=x.y -Dpackaging=jar -Dfile=foobar-x.y.jar*

* Once it is deployed, go into [http://repository.apache.org/](http://repository.apache.org/) by using your SVN
credentials to login in
* Select Staging
* Find your repository artifacts
* Close them (this makes them publicly available, since you are closing the
staging repo)
* Promote them. This adds them to the public Maven repo.
