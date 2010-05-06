/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier;

/**
 * Class containing sample docs from ASF websites under mahout, lucene and spamassasin projects
 *
 */
public final class ClassifierData {
  
  public static final String[][] DATA = {
    {
      "mahout",
      "Mahout's goal is to build scalable machine learning libraries. With scalable we mean: "
         + "Scalable to reasonably large data sets. Our core algorithms for clustering,"
         + " classfication and batch based collaborative filtering are implemented on top "
         + "of Apache Hadoop using the map/reduce paradigm. However we do not restrict "
         + "contributions to Hadoop based implementations: Contributions that run on"},
    {
      "mahout",
      " a single node or on a non-Hadoop cluster are welcome as well. The core"
         + " libraries are highly optimized to allow for good performance also for"
         + " non-distributed algorithms. Scalable to support your business case. "
         + "Mahout is distributed under a commercially friendly Apache Software license. "
         + "Scalable community. The goal of Mahout is to build a vibrant, responsive, "},
    {
      "mahout",
      "diverse community to facilitate discussions not only on the project itself"
         + " but also on potential use cases. Come to the mailing lists to find out more."
         + " Currently Mahout supports mainly four use cases: Recommendation mining takes "
         + "users' behavior and from that tries to find items users might like. Clustering "},
    {
      "mahout",
      "takes e.g. text documents and groups them into groups of topically related documents."
         + " Classification learns from exisiting categorized documents what documents of"
         + " a specific category look like and is able to assign unlabelled documents to "
         + "the (hopefully) correct category. Frequent itemset mining takes a set of item"
         + " groups (terms in a query session, shopping cart content) and identifies, which"
         + " individual items usually appear together."},
    {
      "lucene",
      "Apache Lucene is a high-performance, full-featured text search engine library"
         + " written entirely in Java. It is a technology suitable for nearly any application "
         + "that requires full-text search, especially cross-platform. Apache Lucene is an open source"
         + " project available for free download. Please use the links on the left to access Lucene. "
         + "The new version is mostly a cleanup release without any new features. "},
    {
      "lucene",
      "All deprecations targeted to be removed in version 3.0 were removed. If you "
         + "are upgrading from version 2.9.1 of Lucene, you have to fix all deprecation warnings"
         + " in your code base to be able to recompile against this version. This is the first Lucene"},
    {
      "lucene",
      " release with Java 5 as a minimum requirement. The API was cleaned up to make use of Java 5's "
         + "generics, varargs, enums, and autoboxing. New users of Lucene are advised to use this version "
         + "for new developments, because it has a clean, type safe new API. Upgrading users can now remove"},
    {
      "lucene",
      " unnecessary casts and add generics to their code, too. If you have not upgraded your installation "
         + "to Java 5, please read the file JRE_VERSION_MIGRATION.txt (please note that this is not related to"
         + " Lucene 3.0, it will also happen with any previous release when you upgrade your Java environment)."},
    {
      "spamassasin",
      "SpamAssassin is a mail filter to identify spam. It is an intelligent email filter which uses a diverse "
         + "range of tests to identify unsolicited bulk email, more commonly known as Spam. These tests are applied "
         + "to email headers and content to classify email using advanced statistical methods. In addition, "},
    {
      "spamassasin",
      "SpamAssassin has a modular architecture that allows other technologies to be quickly wielded against spam"
         + " and is designed for easy integration into virtually any email system."
         + "SpamAssassin's practical multi-technique approach, modularity, and extensibility continue to give it an "},
    {
      "spamassasin",
      "advantage over other anti-spam systems. Due to these advantages, SpamAssassin is widely used in all aspects "
         + "of email management. You can readily find SpamAssassin in use in both email clients and servers, on many "
         + "different operating systems, filtering incoming as well as outgoing email, and implementing a "
         + "very broad range "},
    {
      "spamassasin",
      "of policy actions. These installations include service providers, businesses, not-for-profit and "
         + "educational organizations, and end-user systems. SpamAssassin also forms the basis for numerous "
         + "commercial anti-spam products available on the market today."}}; 


  private ClassifierData() { }

}
