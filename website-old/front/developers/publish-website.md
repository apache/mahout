---
layout: default
title: How to Publish Website
theme: 
    name: mahout2
---

## Publishing Docs

This is the most common use case as new docs are published at least with every version release.  They are also updated 
much more frequently as new algorithms and other features are added. 

#### Step 1. Checkout current website from Subversion. 

At the terminal: 

    svn co https://svn.apache.org/repos/asf/mahout asf-mahout
    

#### Step 2. Check the Version you're building

Open `mahout/website/docs/_config.yml`

Edit the `BASE_PATH` variable (near line 59), set this to the version you wish to publish.


#### Step 3. Build Website:
    
    cd $MAHOUT_HOME/website/docs
    JEKYLL_ENV=production bundle exec jekyll build


#### Step 4. Build Scala / Java Docs 
    
From the mahout source top level directory, use maven to build Scala and Java Docs, and copy them to the `mahout/website/docs/_site`

    cd _site
    mkdir scaladocs
    cd ../../../
    
    cd math-scala
    mvn scala:doc
    cp target/site/scaladocs ../website/docs/_site/scaladocs/math-scala -r
    
    cd ../spark
    mvn scala:doc
    cp target/site/scaladocs ../website/docs/_site/scaladocs/spark -r


To build javadocs, make sure you are on Java 7 (`sudo update-alternatives --config java`)

    cd ..
    mvn javadoc:aggregate
    cp target/site/apidocs website/docs/_site/javadocs -r
    
    
If you changed the version number in Step 2, be sure to edit `website/docs/_includes/navbar.html` with a new version. 
    
#### Step 5. Copy `_site` to Subversion

From Mahout top level directory

Delete old version if exists

    rm ../asf-mahout/site/mahout_cms/trunk/content/docs/<version> -r
    cp website/docs/_site ../asf-mahout/site/mahout_cms/trunk/content/docs/<version> -r

Where **<version>** is the version you set in Step 2. 

#### Step 6. Publish Site

    cd /path/to/asf-mahout
    svn add site/mahout_cms/trunk/content/docs/<version>
    svn commit
    
This will "publish" to http://mahout.staging.apache.org Now would be a good time to go do some QA (quality assurance) load 
up the site and check that your links works, esp in the area you were working on.

When you're ready to publish, go to https://cms.apache.org/mahout/ and click publish.

## Publishing Front Site


#### Step 1. Checkout current website from Subversion. 

At the terminal: 

    svn co https://svn.apache.org/repos/asf/mahout asf-mahout
    

#### Step 2. Build Website:
    

    cd $MAHOUT_HOME/website/front
    JEKYLL_ENV=production bundle exec jekyll build


#### Step 3. Copy `_site` to Subversion

From Mahout top level directory

    cp website/front/_site/. ../asf-mahout/site/mahout_cms/trunk/content/ -r



#### Step 6. Publish Site

    cd /path/to/asf-mahout
    svn status | grep '?' | sed 's/^.* /svn add /' | bash
    svn commit
    
    
`svn status | grep '?' | sed 's/^.* /svn add /' | bash` is a clever trick that will pick up any new files. It's the equivelent
to `git add --all` use with care.  You could also manually add files with `svn add path/to/file`


This will "publish" to http://mahout.staging.apache.org Now would be a good time to go do some QA (quality assurance) load 
up the site and check that your links works, esp in the area you were working on.

When you're ready to publish, go to https://cms.apache.org/mahout/ and click publish.

    
    