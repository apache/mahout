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
import java.awt.image.WritableRaster;
import java.io.File;
import java.util.List;

import javax.imageio.ImageIO;

/**
 * @author chimpler.com 
 */
public class ComputeEigenFaces {	
	private static void writeEigenFaceImage(String filename, double[][] eigenFacePixels,
			int width, int height, int columnIndex) throws Exception {
		BufferedImage meanImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		WritableRaster raster = meanImage.getRaster();
		
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for(int i = 0 ; i < eigenFacePixels.length ; i++) {
			min = Math.min(min, eigenFacePixels[i][columnIndex]);
			max = Math.max(max, eigenFacePixels[i][columnIndex]);
		}

		int[] pixels = new int[eigenFacePixels.length];
		for(int i = 0 ; i < eigenFacePixels.length ; i++) {
			pixels[i] = (int)(255.0 * (eigenFacePixels[i][columnIndex] - min) / (max - min));
		}
		raster.setPixels(0, 0, width, height, pixels);

		ImageIO.write(meanImage, "gif", new File(filename));
	}

	private static double[][] computeEigenFaces(double[][] diffMatrix, double[][] eigenVectors) {
		int pixelCount = diffMatrix.length;
		int imageCount = eigenVectors[0].length;
		int rank = eigenVectors.length;
		double[][] eigenFaces = new double[pixelCount][rank];
		
		for(int i = 0 ; i < rank ; i++) {
			double sumSquare = 0;
			for(int j = 0 ; j < pixelCount ; j++) {
				for(int k = 0 ; k < imageCount ; k++) {
					eigenFaces[j][i] += diffMatrix[j][k] * eigenVectors[i][k];
				}
				sumSquare += eigenFaces[j][i] * eigenFaces[j][i]; 
			}
			double norm = Math.sqrt(sumSquare);
			for(int j = 0 ; j < pixelCount ; j++) {
				eigenFaces[j][i] /= norm;
			}
		}
		return eigenFaces;
	}
	
	public static void main(String args[]) throws Exception {
		if (args.length != 7) {
			System.out.println("Arguments: eigenVectorFileName diffMatrixFileName meanImageFileName width height trainingDirectory outputDirectory");
			System.exit(1);
		}

		String eigenVectorsFileName = args[0];
		String diffMatrixFileName = args[1];
		String meanImageFilename = args[2];
		
		int width = Integer.parseInt(args[3]);
		int height = Integer.parseInt(args[4]);
		String trainingDirectory = args[5];
		String outputDirectory = args[6];
		
		File outputDirectoryFile = new File(outputDirectory);
		if (!outputDirectoryFile.exists()) {
			outputDirectoryFile.mkdir();
		}
		
		double[] meanPixels = Helper.readImagePixels(meanImageFilename, width, height);
		double[][] eigenVectors = Helper.readMatrixSequenceFile(eigenVectorsFileName);
		double[][] diffMatrix = Helper.readMatrixSequenceFile(diffMatrixFileName);
		double[][] eigenFaces = computeEigenFaces(diffMatrix, eigenVectors);
		
		int rank = eigenVectors.length;
		for(int i = 0 ; i < rank ; i++) {
			writeEigenFaceImage(outputDirectory + "/eigenface-" + i + ".gif", eigenFaces, width, height, i);
		}

		double minDistance = Double.MAX_VALUE;
		double maxDistance = -Double.MAX_VALUE;
		Helper.writeMatrixSequenceFile(outputDirectory + "/eigenfaces.seq", eigenFaces);

		List<String> imageFileNames = Helper.listImageFileNames(trainingDirectory);
		int imageCount = diffMatrix[0].length;
		int pixelCount = width * height;
		double[][] weightMatrix = new double[imageCount][];
		for(int i = 0 ; i < imageCount ; i++) {
			double[] diffImagePixels = new double[pixelCount];
			for(int j = 0 ; j < pixelCount ; j++) {
				diffImagePixels[j] = diffMatrix[j][i];
			}
			double[] weights = Helper.computeWeights(diffImagePixels, eigenFaces);
			double[] reconstructedImagePixels = Helper.reconstructImageWithEigenFaces(
				weights, eigenFaces, meanPixels);
			String shortFileName = Helper.getShortFileName(imageFileNames.get(i));
			Helper.writeImage(outputDirectory + "/ef-" + shortFileName, reconstructedImagePixels, width, height);

			double[] imagePixels = Helper.readImagePixels(imageFileNames.get(i), width, height);
			double distance = Helper.computeImageDistance(imagePixels, reconstructedImagePixels);
			minDistance = Math.min(minDistance, distance);
			maxDistance = Math.max(maxDistance, distance);
			System.out.printf("Reconstructed Image distance for %1$s: %2$f\n", shortFileName, distance);
			
			weightMatrix[i] = weights;
		}
		Helper.writeMatrixSequenceFile(outputDirectory + "/weights.seq", weightMatrix);
		System.out.println("Min distance = " + minDistance);
		System.out.println("Max distance = " + maxDistance);
	}
	
}
