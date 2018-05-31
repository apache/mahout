

### New Module Structure

```
mahout
- core (contains former mahout-math, mahout-math-scala)
- engines
\- hdfs support
\- abstract (?)
\- spark
\- tensorflow(?)
- community (not tested/built under normal build conditions)
\- flink-batch
\- h2o
\- mr
\- spark_1.6(?)
\- Spark CLI Drivers
- examples
```



#### Todo

-[x] get `org.apache.mahout.math.drm` squared away
-[ ] see `core/pom.xml`
-[ ] move tests for `org.apache.mahout.math.solver`s
-[x] move `org.apache.mahout.math.stats` esp OnlineSolver (but prob all of them)
-[x] move above's tests over.
-[x] IO Tests move over
-[x] Add drivers to Community (to avoid future `scopt` hold ups)
-[ ] update all poms to dump files in `lib/`
-[ ] Move MR to community engines.
-[ ] failing tests on Naivebayes (needs 'Online Summarizer')
