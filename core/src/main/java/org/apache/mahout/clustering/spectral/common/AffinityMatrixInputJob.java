package org.apache.mahout.clustering.spectral.common;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.spectral.eigencuts.EigencutsKeys;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

public final class AffinityMatrixInputJob {

  /**
   * Initializes and executes the job of reading the documents containing
   * the data of the affinity matrix in (x_i, x_j, value) format.
   * 
   * @param input
   * @param output
   * @param rows
   * @param cols
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static void runJob(Path input, Path output, int rows, int cols)
    throws IOException, InterruptedException, ClassNotFoundException {
    HadoopUtil.overwriteOutput(output);

    Configuration conf = new Configuration();
    conf.setInt(EigencutsKeys.AFFINITY_DIMENSIONS, rows);
    Job job = new Job(conf, "AffinityMatrixInputJob: " + input + " -> M/R -> " + output);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(DistributedRowMatrix.MatrixEntryWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(AffinityMatrixInputMapper.class);   
    job.setReducerClass(AffinityMatrixInputReducer.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);

    job.waitForCompletion(true);
  }

  /**
   * A transparent wrapper for the above method which handles the tedious tasks
   * of setting and retrieving system Paths. Hands back a fully-populated
   * and initialized DistributedRowMatrix.
   * @param input
   * @param output
   * @param dimensions
   * @return
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix runJob(Path input, Path output, int dimensions)
    throws IOException, InterruptedException, ClassNotFoundException {
    Path seqFiles = new Path(output, "seqfiles-" + (System.nanoTime() & 0xFF));
    AffinityMatrixInputJob.runJob(input, seqFiles, dimensions, dimensions);
    DistributedRowMatrix A = new DistributedRowMatrix(seqFiles, 
        new Path(seqFiles, "seqtmp-" + (System.nanoTime() & 0xFF)), 
        dimensions, dimensions);
    A.configure(new JobConf());
    return A;
  }
}
