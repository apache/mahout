---
title: "Collaborative Filtering with Implicit Feedback for Scalable Recommendation Systems"
authors: ["Y. Hu", "Y. Koren", "C. Volinsky"]
year: 2008
link: "https://ieeexplore.ieee.org/document/4781121"
---

### Summary
This paper introduces a matrix factorization approach optimized for recommendation systems built from implicit feedback (views, clicks, purchases). Instead of explicit ratings, user interactions are converted into confidence-weighted preferences. The method scales efficiently to millions of users and items, making it foundational for modern recommender engines. The ideas in this paper strongly influenced Mahout’s early CF implementations and continue to guide implicit-feedback recommender design today.