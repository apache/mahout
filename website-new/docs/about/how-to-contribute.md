---
title: How To Contribute
sidebar_label: How To Contribute
---

# How to contribute

*Contributing to an Apache project* is about more than just writing code --
it's about doing what you can to make the project better.  There are lots
of ways to contribute!

<a name="HowToContribute-BeInvolved"></a>
## Get Involved

Discussions at Apache happen on the mailing list. To get involved, you should join the [Mahout mailing lists](/docs/community/mailing-lists).  In particular:

* The **user list** (to help others)
* The **development list** (to join discussions of changes)  -- This is the best place
to understand where the project is headed.
* The **commit list** (to see changes as they are made)

Please keep discussions about Mahout on list so that everyone benefits.
Emailing individual committers with questions about specific Mahout issues
is discouraged.  See [http://people.apache.org/~hossman/#private_q](http://people.apache.org/~hossman/#private_q)
.  Apache  has a number of [email tips for contributors][1] as well.

<a name="HowToContribute-WhattoWorkOn?"></a>
## What to Work On?

What do you like to work on?  There are a ton of things in Mahout that we
would love to have contributions for: documentation, performance improvements, better tests, etc.
The best place to start is by looking into our [issue tracker](https://issues.apache.org/jira/browse/MAHOUT) and
seeing what bugs have been reported and seeing if any look like you could
take them on.  Small, well written, well tested patches are a great way to
get your feet wet.  It could be something as simple as fixing a typo.  The
more important piece is you are showing you understand the necessary steps
for making changes to the code.  Mahout is a pretty big beast at this
point, so changes, especially from non-committers, need to be evolutionary
not revolutionary since it is often very difficult to evaluate the merits
of a very large patch.	Think small, at least to start!

Beyond JIRA, hang out on the dev@ mailing list. That's where we discuss
what we are working on in the internals and where you can get a sense of
where people are working.

Also, documentation is a great way to familiarize yourself with the code
and is always a welcome addition to the codebase and this website. Feel free
to contribute texts and tutorials! Committers will make sure they are added
to this website, and we have a [guide for making website updates][2].
We also have a [wide variety of books and slides][3] for learning more about
machine learning algorithms.

If you are interested in working towards being a committer, general guidelines are available in the [Apache Community documentation](https://community.apache.org/contributors/).

<a name="HowToContribute-ContributingCode(Features,BigFixes,Tests,etc...)"></a>
## Contributing Code (Features, Big Fixes, Tests, etc...)

This section identifies the ''optimal'' steps community member can take to
submit a changes or additions to the Mahout code base.	This can be new
features, bug fixes optimizations of existing features, or tests of
existing code to prove it works as advertised (and to make it more robust
against possible future changes).

Please note that these are the "optimal" steps, and community members that
don't have the time or resources to do everything outlined on this below
should not be discouraged from submitting their ideas "as is" per "Yonik
Seeley's (Solr committer) Law of Patches":

*A half-baked patch in Jira, with no documentation, no tests and no backwards compatibility is better than no patch at all.*

Just because you may not have the time to write unit tests, or cleanup
backwards compatibility issues, or add documentation, doesn't mean other
people don't. Putting your patch out there allows other people to try it
and possibly improve it.

<a name="HowToContribute-Gettingthesourcecode"></a>
## Getting the source code

First of all, you need to get the Mahout source code from [GitHub](https://github.com/apache/mahout). Most development is done on the "main" branch. The first step to making a contribution is to fork Mahout's main branch to your GitHub repository.


<a name="HowToContribute-MakingChanges"></a>
## Making Changes

Before you start, you should send a message to the [Mahout developer mailing list](/docs/community/mailing-lists)
(note: you have to subscribe before you can post), or file a ticket in our [issue tracker](https://issues.apache.org/jira/browse/MAHOUT).
Describe your proposed changes and check that they fit in with what others are doing and have planned for the project.  Be patient, it may take folks a while to understand your requirements.

 1. Create a JIRA Issue in the [issue tracker](https://issues.apache.org/jira/browse/MAHOUT) (if one does not already exist)
 2. Pull the code from your GitHub repository
 3. Ensure that you are working with the latest code from the [apache/mahout](https://github.com/apache/mahout) main branch.
 3. Modify the source code and add some (very) nice features.
     - Be sure to adhere to the following points:
         - All public classes and methods should have informative Javadoc
    comments.
         - Code should be formatted according to standard
    [Java coding conventions](http://www.oracle.com/technetwork/java/codeconventions-150003.pdf),
    with two exceptions:
             - indent two spaces per level, not four.
             - lines can be 120 characters, not 80.
         - Contributions should pass existing unit tests.
         - New unit tests should be provided to demonstrate bugs and fixes.
 4. Commit the changes to your local repository.
 4. Push the code back up to your GitHub repository.
 5. Create a [Pull Request](https://help.github.com/articles/creating-a-pull-request) to the to apache/mahout repository on Github.
     - Include the corresponding JIRA Issue number and description in the title of the pull request:
        - ie. MAHOUT-xxxx: < JIRA-Issue-Description >
 6. Committers and other members of the Mahout community can then comment on the Pull Request.  Be sure to watch for comments, respond and make any necessary changes.

Please be patient. Committers are busy people too. If no one responds to your Pull Request after a few days, please make friendly reminders on the mailing list.  Please
incorporate other's suggestions into into your changes if you think they're reasonable.  Finally, remember that even changes that are not committed are useful to the community.

<a name="HowToContribute-UnitTests"></a>
#### Unit Tests

Please make sure that all unit tests succeed before creating your Pull Request.

Run *mvn clean test*, if you see *BUILD SUCCESSFUL* after the tests have finished, all is ok, but if you see *BUILD FAILED*,
please carefully read the errors messages and check your code.

#### Do's and Don'ts

Please do not:

* reformat code unrelated to the bug being fixed: formatting changes should
be done in separate issues.
* comment out code that is now obsolete: just remove it.
* insert comments around each change, marking the change: folks can use
subversion to figure out what's changed and by whom.
* make things public which are not required by end users.

Please do:

* try to adhere to the coding style of files you edit;
* comment code whose function or rationale is not obvious;
* update documentation (e.g., ''package.html'' files, the website, etc.)


<a name="HowToContribute-Review/ImproveExistingPatches"></a>
## Review/Improve Existing Pull Requests

If there's a JIRA issue that already has a Pull Request with changes that you think are really good, and works well for you -- please add a comment saying so.   If there's room
for improvement (more tests, better javadocs, etc...) then make the changes on your GitHub branch and add a comment about them.	If a lot of people review a Pull Request and give it a
thumbs up, that's a good sign for committers when deciding if it's worth spending time to review it -- and if other people have already put in
effort to improve the docs/tests for an issue, that helps even more.

For more information see [Handling GitHub PRs](http://mahout.apache.org/documentation/developers/github).


  [1]: http://www.apache.org/dev/contrib-email-tips
  [2]: http://mahout.apache.org/documentation/developers/how-to-update-the-website.html
  [3]: http://mahout.apache.org/general/books-tutorials-and-talks.html
