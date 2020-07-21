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
title: Patch Check List

    
---

# Patch Check List

So, you want to merge a contribution- or you want to know in more detail what committers look for in your contribution?
Here are tips, traps, etc. for dealing with
PRs:

  - Did the author write unit tests?  Are the unit tests worthwhile?
  - Are the licenses correct on newly added files? Has an ASF license been
granted?
  - Make sure you update JIRA by assigning the issue to you so that others
know you are working on it.
  - How is the documentation, especially the javadocs?
  - Before committing, make sure you add any new documents to your local Git repo.  
  - Run all unit tests, verify that all tests pass.
  - Lastly, update the [CHANGELOG](https://github.com/apache/mahout/blob/master/CHANGELOG) file. Give proper credit to the authors.
 
After the above steps are verified and completed, and the contribution is ready to merge, follow the steps in the "Merging a PR" section in: [Handling Github PRs](http://mahout.apache.org/developers/github.html).

 - Remember to update the issue status in JIRA when you have completed it.


  