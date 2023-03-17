---
layout: page
title:  News
---

# News

Welcome to the News page! Stay up-to-date with the latest announcements, releases, events, and community highlights from
the Apache Mahout project. Keep an eye on this page for regular updates and make sure you don't miss any important news 
related to the project.

{% for post in site.posts limit:10 %}
- [{{post.title}}]({{ post.url }}) - {{ post.date | date: "%B %d, %Y" }}
{% endfor %}

