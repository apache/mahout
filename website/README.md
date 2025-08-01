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

 # The below are the steps to Setup the Apache Mahout Project Locally.

# 1.Clone the repository 
```bash
git clone https://github.com/username/mahout.git
cd mahout
```
Replace username with your github username.


 
 ## 2. Install Jekyll Locally
 ### A. Prerequisites
 - Ruby Version 2.7.0 or higher
 - RubyGems
 - GCC and Make
## After installing the above pre-requisites, run the following in Terminal
```terminal
gem install jekyll bundler
```
## Once the jekyll is installed, navigate to the website folder of the mahout folder that is cloned in your system

```
cd website
```
## Run the following command while in website directory
```
bundle exec jekyll serve
```
### After running this, you would see the project locally setup at http://localhost:4000/

### Note: If you find any errors running the above command, make sure that jekyll is upto date and ruby is installed on your system 

## To install ruby in your system, run the following command in terminal
```
brew install ruby
```
 # How to post minutes on the home page
1. Make a new GitHub Discussion like https://github.com/apache/mahout/discussions/541
2. Copy the latest Minutes in _posts to a new file, e.g., `cp 2025-05-16-Meeting-Minutes.md 2025-06-13-Meeting-Minutes.md`
3. Edit the new Minutes to reflect the new Discussion link (and update the datestamp and attendees)
4. Add and commit, make a PR
5. The site will automatically build
6. Notify Slack and Twitter
