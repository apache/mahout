

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
\- engines
\-- flink-batch
\-- h2o
\-- spark_1.6(?)
\- mr
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
-[ ] Figure out where mising classes are ^^ OpenIntIntHash, etc.
-[x] failing tests on Naivebayes (needs 'Online Summarizer')
-[ ] Add `examples` module
-[x] ViennaCL to `experimental`
-[ ] Update `.travis.yml` for new module structures
-[ ] Add profiles back in.
-[x] Fix POMs (correct heirarcy / inheritance) (for now is done...)
-[ ] Fix POMs (add required plugins for release)
-[ ] Clean up, delete directories no longer in use.
-[ ] Create profile to create spark-fat-jar. (of if you get pushback to make no fat jars)
-[ ] Move Kryo to top pom (spark, core and flink at least)
-[ ] Inspect all warnings
-[ ] Update Website 
-[ ] - Description of modules
-[ ] - Available profiles and what they do
