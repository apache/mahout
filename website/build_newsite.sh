#/bin/bash

MHT_GIT_DIR=/data/work/git-repos/mahout/
TMP_WEB_DIR=/tmp/new/mahout_tmp_web/
BASEDOCURL="docs/0.13.1-SNAPSHOT"

THEME=mahout # mahout-retro, mahout, mahout2, mahout3

# Checkout master and update to latest code
rm -rf $TMP_WEB_DIR/*
mkdir -p $TMP_WEB_DIR
cd $MHT_GIT_DIR
# git checkout website
# git fetch apache
# git reset --hard apache/master
#

# Build static content
cd ${MHT_GIT_DIR}/website/front
rake theme:switch name="$THEME"
JEKYLL_ENV=production bundle exec jekyll build
cp -R ${MHT_GIT_DIR}/website/front/_site/* $TMP_WEB_DIR


# # Build version specific content
mkdir -p ${TMP_WEB_DIR}/${BASEDOCURL}
rm -rf ${TMP_WEB_DIR}/${BASEDOCURL}/*
cd ${MHT_GIT_DIR}/website/docs
rake theme:switch name="$THEME"
JEKYLL_ENV=production bundle exec jekyll build
cp -R ${MHT_GIT_DIR}/website/docs/_site/* ${TMP_WEB_DIR}/${BASEDOCURL}/
#cp -R ${MHT_GIT_DIR}/website/docs/_site/* ${TMP_WEB_DIR}/0.13.0/

echo "-------------------------------------------------------"
echo To test website open a new with browser window at localhost:4000 
echo "Start Jekyll server in  a terminal window..."
echo cd $TMP_WEB_DIR
echo jekyll serve
echo "-------------------------------------------------------"
