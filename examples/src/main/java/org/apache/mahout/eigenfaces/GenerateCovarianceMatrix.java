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

import java.io.File;
import java.util.Collection;
import java.util.List;

/**
 * @author chimpler.com 
 */
public class GenerateCovarianceMatrix {
	
	private static double[][] convertImagesToMatrix(Collection<String> imageFileNames,
			int width, int height) throws Exception {
		int columnIndex = 0;
		double[][] pixelMatrix = new double[width * height][imageFileNames.size()];
		for(String fileName: imageFileNames) {
			System.out.println("Reading file " + fileName);
			double[] pixels = Helper.readImagePixels(fileName, width, height);
			for(int i = 0 ; i < pixels.length ; i++) {
				pixelMatrix[i][columnIndex] = pixels[i];
			}
			columnIndex++;
		}

		return pixelMatrix;
	}
	
	private static double[] computeMeanColumn(double[][] pixelMatrix) {
		int pixelCount = pixelMatrix.length;
		double[] meanColumn = new double[pixelCount];
		int columnCount = pixelMatrix[0].length;
		for(int i = 0 ; i < pixelCount ; i++) {
			int sum = 0;
			for(int j = 0 ; j < columnCount ; j++) {
				sum += pixelMatrix[i][j];
			}
			meanColumn[i] = sum / columnCount;
		}
		return meanColumn;		
	}
	
	private static double[][] computeCovarianceMatrix(double[][] diffMatrixPixels) {
		int rowCount = diffMatrixPixels.length;
		int columnCount = diffMatrixPixels[0].length;
		
		double[][] covarianceMatrix = new double[columnCount][columnCount];
		for(int i = 0 ; i < columnCount ; i++) {
			for(int j = 0 ; j < columnCount ; j++) {
				int sum = 0;
				for(int k = 0 ; k < rowCount ; k++) {
					sum += diffMatrixPixels[k][i] * diffMatrixPixels[k][j];
				}
				covarianceMatrix[i][j] = sum;
			}
		}
		
		return covarianceMatrix;
	}


	public static void main(String args[]) throws Exception {
		if (args.length != 4) {
			System.out.println("Arguments: width height trainingDirectory outputDirectory");
			System.exit(1);
		}

		int width = Integer.parseInt(args[0]);
		int height = Integer.parseInt(args[1]);
		String imageDirectory = args[2];
		String outputDirectory = args[3];

		File outputDirectoryFile = new File(outputDirectory);
		if (!outputDirectoryFile.exists()) {
			outputDirectoryFile.mkdir();
		}
		
		List<String> imageFileNames = Helper.listImageFileNames(imageDirectory);

		System.out.println("Reading " + imageFileNames.size() + " images...");
		double[][] pixelMatrix = convertImagesToMatrix(imageFileNames, width, height);
		double[] meanColumn = computeMeanColumn(pixelMatrix);
		Helper.writeImage(outputDirectory + "/mean-image.gif", meanColumn, width, height);
		double[][] diffMatrixPixels = Helper.computeDifferenceMatrixPixels(pixelMatrix, meanColumn);
		Helper.writeMatrixSequenceFile(outputDirectory + "/diffmatrix.seq", diffMatrixPixels);
		
		double[][] covarianceMatrix = computeCovarianceMatrix(diffMatrixPixels);
		Helper.writeMatrixSequenceFile(outputDirectory + "/covariance.seq", covarianceMatrix);
	}
}
