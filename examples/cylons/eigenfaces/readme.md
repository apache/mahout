
Requires Mahout 0.13.1-SNAPSHOT, with Scala 2.11, Viennacl, Spark 2.1
(in theory, Mahout 0.13.0 should work, but you need to change the version property in pom.xml)

	git clone http://github.com/apache/mahout
	cd mahout
	mvn clean install -Pscala-2.11,spark-2.1,viennacl-omp

Presuming you've already built the jar...

	export SPARK_HOME=/path/to/your/spark
	$SPARK_HOME/sbin/start-all.sh
	