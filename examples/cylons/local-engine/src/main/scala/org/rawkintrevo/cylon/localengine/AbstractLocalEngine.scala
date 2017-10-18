package org.rawkintrevo.cylon.localengine

import org.apache.mahout.math.Matrix
import org.apache.solr.client.solrj.SolrClient
import org.apache.solr.client.solrj.impl.HttpSolrClient
import org.opencv.core.Core
import org.opencv.videoio.VideoCapture
import org.rawkintrevo.cylon.common.mahout.MahoutUtils
import org.rawkintrevo.cylon.common.solr.CylonSolrClient

import org.slf4j.{Logger, LoggerFactory}

trait AbstractLocalEngine {

  val logger: Logger = LoggerFactory.getLogger(getClass.getName)

  val cylonSolrClient = new CylonSolrClient
  var inputPath: String = _

  def connectToSolr(solrURL: String = "http://localhost:8983/solr/cylonfaces") = {
    logger.info(s"Establishing connectiong to Solr instance at ${solrURL}")
    cylonSolrClient.connect(solrURL)
  }


}
