/*
 * Copyright (c) 2010 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.eigenfaces;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.imageio.ImageIO;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;

/**
 * @author chimpler.com 
 */
public class Helper {
	public static void writeImage(String filename, double[] imagePixels,
			int width, int height) throws Exception {
		BufferedImage meanImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		WritableRaster raster = meanImage.getRaster();
		
		// convert byte array to byte array
		int[] pixels = new int[imagePixels.length];
		for(int i = 0 ; i < imagePixels.length ; i++) {
			pixels[i] = (int)imagePixels[i];
		}
		raster.setPixels(0, 0, width, height, pixels);

		ImageIO.write(meanImage, "gif", new File(filename));
	}

	public static double[] readImagePixels(String imageFileName, int width, int height) throws Exception {
		BufferedImage colorImage = ImageIO.read(new File(imageFileName));
		
		// convert to grayscale image
		BufferedImage greyImage = new BufferedImage(
				width,
				height,
			    BufferedImage.TYPE_BYTE_GRAY);
		greyImage.getGraphics().drawImage(colorImage, 0, 0, width, height, null);
		
		byte[] bytePixels = ((DataBufferByte)greyImage.getRaster().getDataBuffer()).getData();
		
		double[] doublePixels = new double[bytePixels.length];
		for(int i = 0 ; i < doublePixels.length ; i++) {
			doublePixels[i] = (double)(bytePixels[i] & 255);
		}
		return doublePixels;
	}
	
	public static double[][] computeDifferenceMatrixPixels(double[][] matrixPixels, double[] meanColumn) {
		int rowCount = matrixPixels.length;
		int columnCount = matrixPixels[0].length;
		
		double[][] diffMatrixPixels = new double[rowCount][columnCount];
		for(int i = 0 ; i < rowCount ; i++) {
			for(int j = 0 ; j < columnCount ; j++) {
				diffMatrixPixels[i][j] = matrixPixels[i][j] - meanColumn[i];
			}
		}
		
		return diffMatrixPixels;
	}

	public static double[] computeDifferencePixels(double[] pixels, double[] meanColumn) {
		int pixelCount = pixels.length;
		double[] diffPixels = new double[pixelCount];
		for(int i = 0 ; i < pixelCount ; i++) {
			diffPixels[i] = pixels[i] - meanColumn[i];
		}
		
		return diffPixels;
	}
	
	public static double[][] readMatrixSequenceFile(String fileName) throws Exception {
		Configuration configuration = new Configuration();
		FileSystem fs = FileSystem.get(configuration);
		Reader matrixReader = new SequenceFile.Reader(fs, 
			new Path(fileName), configuration);
		
		List<double[]> rows = new ArrayList<double[]>();
		IntWritable key = new IntWritable();
		VectorWritable value = new VectorWritable();
		while(matrixReader.next(key, value)) {
			Vector vector = value.get();
			double[] row = new double[vector.size()];
			for(int i = 0 ; i < vector.getNumNondefaultElements() ; i++) {
				Element element = vector.getElement(i);
				row[element.index()] = element.get();
			}
			rows.add(row);
		}
		return rows.toArray(new double[rows.size()][]);
	}

	public static void writeMatrixSequenceFile(String matrixSeqFileName, double[][] covarianceMatrix) throws Exception{
		int rowCount = covarianceMatrix.length;
		int columnCount = covarianceMatrix[0].length;

		Configuration configuration = new Configuration();
		FileSystem fs = FileSystem.get(configuration);
		Writer matrixWriter = new SequenceFile.Writer(fs, configuration, 
				new Path(matrixSeqFileName),
				IntWritable.class, VectorWritable.class);

		IntWritable key = new IntWritable();
		VectorWritable value = new VectorWritable();
		
		double[] doubleValues = new double[columnCount];
		for(int i = 0 ; i < rowCount ; i++) {
			key.set(i);			
			for(int j = 0 ; j < columnCount ; j++) {
				doubleValues[j] = covarianceMatrix[i][j];
			}
			Vector vector = new DenseVector(doubleValues);
			value.set(vector);
			
			matrixWriter.append(key, value);
		}
		matrixWriter.close();
	}

	public static double[] computeWeights(double[] diffImagePixels,
			double[][] eigenFaces) {
		int pixelCount = eigenFaces.length;
		int eigenFaceCount = eigenFaces[0].length;
		double[] weights = new double[eigenFaceCount];
		for(int i = 0 ; i < eigenFaceCount ; i++) {
			for(int j = 0 ; j < pixelCount ; j++) {
				weights[i] += diffImagePixels[j] * eigenFaces[j][i];
			}
		}
		return weights;
	}

	public static double[] reconstructImageWithEigenFaces(
			double[] weights,
			double[][] eigenFaces, 
			double[] meanImagePixels) throws Exception {
		int pixelCount = eigenFaces.length;
		int eigenFaceCount = eigenFaces[0].length;
		
		// reconstruct image from weight and eigenfaces
		double[] reconstructedPixels = new double[pixelCount];
		for(int i = 0 ; i < eigenFaceCount ; i++) {
			for(int j = 0 ; j < pixelCount ; j++) {
				reconstructedPixels[j] += weights[i] * eigenFaces[j][i];
			}
		}

		// add mean
		for(int i = 0 ; i < pixelCount ; i++) {
			reconstructedPixels[i] += meanImagePixels[i];
		}

		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for(int i = 0 ; i < reconstructedPixels.length ; i++) {
			min = Math.min(min, reconstructedPixels[i]);
			max = Math.max(max, reconstructedPixels[i]);
		}

		double[] normalizedReconstructedPixels = new double[pixelCount];
		for(int i = 0 ; i < reconstructedPixels.length ; i++) {
			normalizedReconstructedPixels[i] = (255.0 * (reconstructedPixels[i] - min)) / (max - min);
		}
		
		return normalizedReconstructedPixels;
	}

	public static double computeImageDistance(double[] pixelImage1, double[] pixelImage2) {
		double distance = 0;
		int pixelCount = pixelImage1.length;
		for(int i = 0 ; i < pixelCount ; i++) {
			double diff = pixelImage1[i] - pixelImage2[i];
			distance += diff * diff;
		}
		return Math.sqrt(distance / pixelCount);
	}

	public static List<String> listImageFileNames(String directoryName) {
		File directory = new File(directoryName);
		List<String> imageFileNames = new ArrayList<String>();
		for(File imageFile: directory.listFiles()) {
//			if (imageFile.getName().endsWith(".gif")) {
				imageFileNames.add(imageFile.getAbsolutePath());
//			}
		}			
		Collections.sort(imageFileNames);
		return imageFileNames;
	}
	
	public static String getShortFileName(String fullFileName) {
		return new File(fullFileName).getName();
	}
}
