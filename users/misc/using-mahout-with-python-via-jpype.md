<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
---
layout: default
title: Using Mahout with Python via JPype

    
---

<a name="UsingMahoutwithPythonviaJPype-overview"></a>
# Mahout over Jython - some examples
This tutorial provides some sample code illustrating how we can read and
write sequence files containing Mahout vectors from Python using JPype.
This tutorial is intended for people who want to use Python for analyzing
and plotting Mahout data. Using Mahout from Python turns out to be quite
easy.

This tutorial concerns the use of cPython (cython) as opposed to Jython.
JPython wasn't an option for me, because  (to the best of my knowledge)
JPython doesn't work with Python extensions numpy, matplotlib, or h5py
which I rely on heavily.

The instructions below explain how to setup a python script to read and
write the output of Mahout clustering.

You will first need to download and install the JPype package for python.

The first step to setting up JPype is determining the path to the dynamic
library for the jvm ; on linux this will be a .so file on and on windows it
will be a .dll.

In your python script, create a global variable with the path to this dll



Next we need to figure out how we need to set the classpath for mahout. The
easiest way to do this is to edit the script in "bin/mahout" to print out
the classpath. Add the line "echo $CLASSPATH" to the script somewhere after
the comment "run it" (this is line 195 or so). Execute the script to print
out the classpath.  Copy this output and paste it into a variable in your
python script. The result for me looks like the following




Now we can create a function to start the jvm in python using jype

    jvm=None
    def start_jpype():
    global jvm
    if (jvm is None):
    cpopt="-Djava.class.path={cp}".format(cp=classpath)
    startJVM(jvmlib,"-ea",cpopt)
    jvm="started"



<a name="UsingMahoutwithPythonviaJPype-WritingNamedVectorstoSequenceFilesfromPython"></a>
# Writing Named Vectors to Sequence Files from Python
We can now use JPype to create sequence files which will contain vectors to
be used by Mahout for kmeans. The example below is a function which creates
vectors from two Gaussian distributions with unit variance.


    def create_inputs(ifile,*args,**param):
     """Create a sequence file containing some normally distributed
    	ifile - path to the sequence file to create
     """
     
     #matrix of the cluster means
     cmeans=np.array([[1,1] ,[-1,-1]],np.int)
     
     nperc=30  #number of points per cluster
     
     vecs=[]
     
     vnames=[]
     for cind in range(cmeans.shape[0]):
      pts=np.random.randn(nperc,2)
      pts=pts+cmeans[cind,:].reshape([1,cmeans.shape[1]])
      vecs.append(pts)
     
      #names for the vectors
      #names are just the points with an index
      #we do this so we can validate by cross-refencing the name with thevector
      vn=np.empty(nperc,dtype=(np.str,30))
      for row in range(nperc):
       vn[row]="c"+str(cind)+"_"+pts[row,0].astype((np.str,4))+"_"+pts[row,1].astype((np.str,4))
      vnames.append(vn)
      
     vecs=np.vstack(vecs)
     vnames=np.hstack(vnames)
     
    
     #start the jvm
     start_jpype()
     
     #create the sequence file that we will write to
     io=JPackage("org").apache.hadoop.io 
     FileSystemCls=JPackage("org").apache.hadoop.fs.FileSystem
     
     PathCls=JPackage("org").apache.hadoop.fs.Path
     path=PathCls(ifile)
    
     ConfCls=JPackage("org").apache.hadoop.conf.Configuration 
     conf=ConfCls()
     
     fs=FileSystemCls.get(conf)
     
     #vector classes
     VectorWritableCls=JPackage("org").apache.mahout.math.VectorWritable
     DenseVectorCls=JPackage("org").apache.mahout.math.DenseVector
     NamedVectorCls=JPackage("org").apache.mahout.math.NamedVector
     writer=io.SequenceFile.createWriter(fs, conf, path,io.Text,VectorWritableCls)
     
     
     vecwritable=VectorWritableCls()
     for row in range(vecs.shape[0]):
      nvector=NamedVectorCls(DenseVectorCls(JArray(JDouble,1)(vecs[row,:])),vnames[row])
      #need to wrap key and value because of overloading
      wrapkey=JObject(io.Text("key "+str(row)),io.Writable)
      wrapval=JObject(vecwritable,io.Writable)
      
      vecwritable.set(nvector)
      writer.append(wrapkey,wrapval)
      
     writer.close()


<a name="UsingMahoutwithPythonviaJPype-ReadingtheKMeansClusteredPointsfromPython"></a>
# Reading the KMeans Clustered Points from Python
Similarly we can use JPype to easily read the clustered points outputted by
mahout.

    def read_clustered_pts(ifile,*args,**param):
     """Read the clustered points
     ifile - path to the sequence file containing the clustered points
     """ 
    
     #start the jvm
     start_jpype()
     
     #create the sequence file that we will write to
     io=JPackage("org").apache.hadoop.io 
     FileSystemCls=JPackage("org").apache.hadoop.fs.FileSystem
     
     PathCls=JPackage("org").apache.hadoop.fs.Path
     path=PathCls(ifile)
    
     ConfCls=JPackage("org").apache.hadoop.conf.Configuration 
     conf=ConfCls()
     
     fs=FileSystemCls.get(conf)
     
     #vector classes
     VectorWritableCls=JPackage("org").apache.mahout.math.VectorWritable
     NamedVectorCls=JPackage("org").apache.mahout.math.NamedVector
     
     
     ReaderCls=io.__getattribute__("SequenceFile$Reader") 
     reader=ReaderCls(fs, path,conf)
     
    
     key=reader.getKeyClass()()
     
    
     valcls=reader.getValueClass()
     vecwritable=valcls()
     while (reader.next(key,vecwritable)):	
      weight=vecwritable.getWeight()
      nvec=vecwritable.getVector()
      
      cname=nvec.__class__.__name__
      if (cname.rsplit('.',1)[1]=="NamedVector"):  
       print "cluster={key} Name={name} x={x}y={y}".format(key=key.toString(),name=nvec.getName(),x=nvec.get(0),y=nvec.get(1))
      else:
       raise NotImplementedError("Vector isn't a NamedVector. Need tomodify/test the code to handle this case.")


<a name="UsingMahoutwithPythonviaJPype-ReadingtheKMeansCentroids"></a>
# Reading the KMeans Centroids
Finally we can create a function to print out the actual cluster centers
found by mahout,

    def getClusters(ifile,*args,**param):
     """Read the centroids from the clusters outputted by kmenas
    	   ifile - Path to the sequence file containing the centroids
     """ 
    
     #start the jvm
     start_jpype()
     
     #create the sequence file that we will write to
     io=JPackage("org").apache.hadoop.io 
     FileSystemCls=JPackage("org").apache.hadoop.fs.FileSystem
     
     PathCls=JPackage("org").apache.hadoop.fs.Path
     path=PathCls(ifile)
    
     ConfCls=JPackage("org").apache.hadoop.conf.Configuration 
     conf=ConfCls()
     
     fs=FileSystemCls.get(conf)
     
     #vector classes
     VectorWritableCls=JPackage("org").apache.mahout.math.VectorWritable
     NamedVectorCls=JPackage("org").apache.mahout.math.NamedVector
     ReaderCls=io.__getattribute__("SequenceFile$Reader")
     reader=ReaderCls(fs, path,conf)
     
    
     key=io.Text()
     
    
     valcls=reader.getValueClass()
    
     vecwritable=valcls()
     
     while (reader.next(key,vecwritable)):	
      center=vecwritable.getCenter()
      
      print "id={cid}center={center}".format(cid=vecwritable.getId(),center=center.values)
      pass

