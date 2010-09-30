package org.apache.mahout.clustering.spectral.common;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.spectral.kmeans.SpectralKMeansDriver;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

/**
 * <p>Handles reading the files representing the affinity matrix. Since the affinity
 * matrix is representative of a graph, each line in all the files should
 * take the form:</p>
 * 
 * <code>i,j,value</code>
 * 
 * <p>where <code>i</code> and <code>j</code> are the <code>i</code>th and
 * <code>j</code> data points in the entire set, and <code>value</code> 
 * represents some measurement of their relative absolute magnitudes. This
 * is, simply, a method for representing a graph textually.
 */
public class AffinityMatrixInputMapper extends Mapper<LongWritable, Text, IntWritable, DistributedRowMatrix.MatrixEntryWritable> {

	@Override
	protected void map(LongWritable key, Text value, Context context) 
						throws IOException, InterruptedException {
		
		String [] elements = value.toString().split(",");
		if (SpectralKMeansDriver.DEBUG) {
			System.out.println("(DEBUG - MAP) Key[" + key.get() + "], " + 
						"Value[" + value.toString() + "]");
		}
		
		// enforce well-formed textual representation of the graph
		if (elements.length != 3) {
			throw new IOException("Expected input of length 3, received " + 
					elements.length + ". Please make sure you adhere to " + 
					"the structure of (i,j,value) for representing a graph " +
					"in text.");
		} else if (elements[0].length() == 0 || elements[1].length() == 0 || 
				elements[2].length() == 0) {
			throw new IOException("Found an element of 0 length. Please " +
					"be sure you adhere to the structure of (i,j,value) for " +
					"representing a graph in text.");
		}
			
		// parse the line of text into a DistributedRowMatrix entry,
		// making the row (elements[0]) the key to the Reducer, and
		// setting the column (elements[1]) in the entry itself
		DistributedRowMatrix.MatrixEntryWritable toAdd = 
			new DistributedRowMatrix.MatrixEntryWritable();
		IntWritable row = new IntWritable(Integer.valueOf(elements[0]));
		toAdd.setRow(-1); // already set as the Reducer's key 
		toAdd.setCol(Integer.valueOf(elements[1]));
		toAdd.setVal(Double.valueOf(elements[2]));
		context.write(row, toAdd);
	}
}
