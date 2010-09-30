package org.apache.mahout.clustering.spectral.common;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.spectral.eigencuts.EigencutsKeys;
import org.apache.mahout.clustering.spectral.kmeans.SpectralKMeansDriver;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

/**
 * Tasked with taking each DistributedRowMatrix entry and collecting them
 * into vectors corresponding to rows. The input and output keys are the same,
 * corresponding to the row in the ensuing matrix. The matrix entries are
 * entered into a vector according to the column to which they belong, and
 * the vector is then given the key corresponding to its row.
 */
public class AffinityMatrixInputReducer extends
		Reducer<IntWritable, DistributedRowMatrix.MatrixEntryWritable, 
				IntWritable, VectorWritable> {
	
	@Override
	protected void reduce(IntWritable row, 
							Iterable<DistributedRowMatrix.MatrixEntryWritable> values,
							Context context) 
							throws IOException, InterruptedException {
		RandomAccessSparseVector out = new RandomAccessSparseVector(
							context.getConfiguration()
							.getInt(EigencutsKeys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE), 100);

		for (DistributedRowMatrix.MatrixEntryWritable element : values) {
			out.setQuick(element.getCol(), element.getVal());
			if (SpectralKMeansDriver.DEBUG) {
				System.out.println("(DEBUG - REDUCE) Row[" + row.get() + "], " + 
						"Column[" + element.getCol() + "], Value[" + 
						element.getVal() + "]");
			}
		}
		SequentialAccessSparseVector output = new SequentialAccessSparseVector(out);
		context.write(row, new VectorWritable(output));
	}
}
