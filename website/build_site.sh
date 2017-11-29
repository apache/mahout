#!/usr/bin/env bash

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
(cd oldsite && bundle install --path ${GEM_HOME})
(cd oldsite && bundle)
(cd oldsite && bundle exec jekyll build --destination $WORKDIR)
(cd docs && bundle install --path ${GEM_HOME})
(cd docs && bundle)
(cd docs && bundle exec jekyll build --destination $WORKDIR/docs/latest)


# Set env for docs
MAHOUT_VERSION=0.13.0
DISTFILE=apache-mahout-distribution-$MAHOUT_VERSION.tar.gz
DISTPATH=http://mirrors.ocf.berkeley.edu/apache/mahout/$MAHOUT_VERSION/$DISTFILE

# Copy API docs into _site
wget -P $WORKDIR $DISTPATH
tar -C $WORKDIR -xzf $WORKDIR/$DISTFILE apache-mahout-distribution-$MAHOUT_VERSION/docs
mkdir -p $WORKDIR/docs/$MAHOUT_VERSION/api
mv $WORKDIR/apache-mahout-distribution-$MAHOUT_VERSION/docs $WORKDIR/docs/$MAHOUT_VERSION/api
rm -f $WORKDIR/$DISTFILE
git checkout asf-site
git clean -f -d
git pull origin asf-site
rm -rf *
cp -a $WORKDIR/* .
git add .
git commit -m "Automatic Site Publish by Buildbot"
<<<<<<< HEAD
git push origin asf-site
=======
git push origin asf-site
>>>>>>> e591012439c04e98d669ef9732fde865a9ef76fa
