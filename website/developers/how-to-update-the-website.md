---
layout: default
title: How To Update The Website
theme: mahout2
---

# How to update the Mahout Website

<a name="HowToUpdateTheWebsite-Howtoupdatethemahouthomepage"></a>
## How to update the mahout home page
1. Clone Apache Mahout, the website is contained in the `website/` folder, and all pages are writtin in markdown.
1. Once you have made appropriate changes, please open a pull request. 

<a name="HowToUpdateTheWebsite-SomeDo'sandDont'sofupdatingthewiki"></a>
## Some Do's and Dont's of updating the web site
1. Keep all pages cleanly formatted - this includes using standard formatting for headers etc.
1. Try to keep a single page for a topic instead of starting multiple ones.
If the topics are related, put it under as a child under the similar page.
1. Notify the developers of orphaned or broken links.

## How to push changes to the actual website (committers only)

1. `svn co svn co https://svn.apache.org/repos/asf/mahout asf-mahout`

1. Run Terminal
       ```
       bundle exec jekyll build
       ```

1. Copy `mahout/website/_site` to `asf-mahout/site/docs/<MAHOUT-VERSION>/`

1. `svn commit` 

... `svnpubsub` will come into play... 
https://reference.apache.org/committer/website