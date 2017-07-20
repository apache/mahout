#/bin/bash


#################################################
# create OLDSITE 
#################################################

MHT_GIT_DIR=/data/work/git-repos/mahout/
TMP_WEB_DIR=/tmp/mahout_tmp_web/

# Checkout master and update to latest code
rm -rf $TMP_WEB_DIR/*
mkdir -p $TMP_WEB_DIR
cd $MHT_GIT_DIR
# git checkout master
# git fetch apache
# git reset --hard apache/master
#

# Build static content
cd ${MHT_GIT_DIR}/website/oldsite
rake theme:switch name="mahout-retro"
JEKYLL_ENV=production bundle exec jekyll build
cp -R ${MHT_GIT_DIR}/website/oldsite/_site/* $TMP_WEB_DIR

echo "-------------------------------------------------------"
echo To test website open a new with browser window at localhost:4000 
echo "Start Jekyll server in  a terminal window..."
echo cd $TMP_WEB_DIR
echo jekyll serve
echo "-------------------------------------------------------"
