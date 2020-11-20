#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# Big thanks to Project Aria Tosca Incubating! (rawkintrevo lifted/modded their script Nov 29, 2017))

WORKDIR=/tmp/mahout-site
rm -rf $WORKDIR
mkdir -p $WORKDIR/docs/latest

export RUBY_PATH=~/shared/.rvm
export GEM_HOME=${RUBY_PATH}/gems
curl -sSL https://get.rvm.io | bash -s -- --path ${RUBY_PATH}
mkdir -p ${GEM_HOME}/gems
gem install  --install-dir ${GEM_HOME} bundler
export PATH=${GEM_HOME}/bin:$PATH
(cd website && bundle install --path ${GEM_HOME})
(cd website && bundle)
(cd website && bundle exec jekyll build --destination $WORKDIR)

# Set env for docs
MAHOUT_VERSION=0.13.0



git checkout asf-site
git clean -f -d
git pull origin asf-site
if [ ! -d "/docs/$MAHOUT_VERSION/api" ]; then
	echo "API docs for $MAHOUT_VERSION not found, downloading them"
	DISTFILE=apache-mahout-distribution-$MAHOUT_VERSION.tar.gz
	DISTPATH=https://dist.apache.org/repos/dist/release/mahout/$MAHOUT_VERSION/$DISTFILE
	# Copy API docs into _site
	wget -q --directory-prefix=$WORKDIR $DISTPATH
	echo "Unzipping..."
	tar -C $WORKDIR -xzf $WORKDIR/$DISTFILE apache-mahout-distribution-$MAHOUT_VERSION/docs
	mkdir -p $WORKDIR/docs/$MAHOUT_VERSION/api
	echo "Moving directory..."
	mv $WORKDIR/apache-mahout-distribution-$MAHOUT_VERSION/docs $WORKDIR/docs/$MAHOUT_VERSION/api
	rm -f $WORKDIR/$DISTFILE
	echo "done."
fi
# rm -rf *
#cp -a $WORKDIR/* .
#cp -r $WORKDIR/* .
#git add .
#git commit -m "Automatic Site Publish by Buildbot"
#git push origin asf-site

