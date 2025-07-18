---
layout: page
title: Patch Check List
redirect_from:
  - /documentation/developers/patch-check-list
  - /documentation/developers/patch-check-list.html

    
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
 
After the above steps are verified and completed, and the contribution is ready to merge, follow the steps in the "Merging a PR" section in: [Handling Github PRs](github.html).

 - Remember to update the issue status in JIRA when you have completed it.


