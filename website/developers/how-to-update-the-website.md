---
layout: page
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

1. `svn co (link to website repo?)`

1. Run Terminal 1
       ```
       bundle exec jekyll serve
       ```
       
Terminal 2
       ```
       bundle exec jekyll build --watch
       ```
(probably not needed, which ever one builds `_site/`)

1. Copy `_site` to `svn_repo/.../docs/<MAHOUT-VERSION>/`

1. `svn commit` or whatever they use.