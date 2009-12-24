#!/bin/bash

TASK_OPTS=-Dmapred.job.queue.name=grideng\ -Dmapred.map.tasks.speculative.execution=true\ -Dmapred.reduce.tasks.speculative.execution=true
#JAVA_OPTS=-Dmapred.child.java.opts=-Xmx1280m\ -server\ -Djava.net.preferIPv4Stack=true
NETFLIX_IN_DIR=netflix-data
NETFLIX_OUT_DIR=netflix-out
BIGRAMS_OUT_DIR=$NETFLIX_OUT_DIR/bigrams
SIMILAR_MOVIES_OUT_DIR=$NETFLIX_OUT_DIR/similarMovies
USER_MOVIES_JOINED_DIR=$NETFLIX_OUT_DIR/user_movies_joined
RECOMMENDATIONS_DIR=$NETFLIX_OUT_DIR/recommendations
MAX_RECOMMENDATIONS=50
MAX_SIMILAR_MOVIES=100
REDUCERS=200

echo "Generating bigrams for movie similarity computation ..."
hadoop dfs -rmr $BIGRAMS_OUT_DIR
hadoop jar mahout-core-0.3-SNAPSHOT.jar org.apache.mahout.cf.taste.hadoop.cooccurence.ItemBigramGenerator -Dmapred.child.java.opts=-Xmx1280m\ -server\ -Djava.net.preferIPv4Stack=true $TASK_OPTS $NETFLIX_IN_DIR $BIGRAMS_OUT_DIR $REDUCERS
echo "Done."
echo "Computing co-occurrence based movie similarity scores ..."
hadoop dfs -rmr $SIMILAR_MOVIES_OUT_DIR
hadoop jar mahout-core-0.3-SNAPSHOT.jar org.apache.mahout.cf.taste.hadoop.cooccurence.ItemSimilarityEstimator -Dmapred.child.java.opts=-Xmx1280m\ -server\ -Djava.net.preferIPv4Stack=true $TASK_OPTS  $BIGRAMS_OUT_DIR $SIMILAR_MOVIES_OUT_DIR $MAX_SIMILAR_MOVIES $REDUCERS
echo "Done."
echo "Joining User history with similar items ..."
hadoop dfs -rmr $USER_MOVIES_JOINED_DIR
hadoop jar mahout-core-0.3-SNAPSHOT.jar org.apache.mahout.cf.taste.hadoop.cooccurence.UserItemJoiner -Dmapred.child.java.opts=-Xmx1280m\ -server\ -Djava.net.preferIPv4Stack=true $TASK_OPTS $NETFLIX_IN_DIR $SIMILAR_MOVIES_OUT_DIR $USER_MOVIES_JOINED_DIR $REDUCERS
echo "Done."

echo "Generating recommendations now ..."
hadoop dfs -rmr $RECOMMENDATIONS_DIR
hadoop jar mahout-core-0.3-SNAPSHOT.jar org.apache.mahout.cf.taste.hadoop.cooccurence.UserItemRecommender -Dmapred.child.java.opts=-Xmx1280m\ -server\ -Djava.net.preferIPv4Stack=true $TASK_OPTS $USER_MOVIES_JOINED_DIR $RECOMMENDATIONS_DIR $MAX_RECOMMENDATIONS $REDUCERS

