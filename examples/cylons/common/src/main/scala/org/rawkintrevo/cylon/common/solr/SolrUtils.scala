package org.rawkintrevo.cylon.common.solr

import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter

import org.apache.solr.client.solrj.{SolrClient, SolrQuery}
import org.apache.solr.client.solrj.SolrQuery.SortClause
import org.apache.mahout.math.scalabindings.MahoutCollections._
import org.apache.solr.client.solrj.impl.HttpSolrClient
import org.apache.solr.client.solrj.response.QueryResponse
import org.apache.solr.common.{SolrDocument, SolrInputDocument}

class CylonSolrClient extends Serializable{

  var solrClient: SolrClient = _

  def connect(solrURL: String) = {
    solrClient = new HttpSolrClient.Builder(solrURL).build()
  }

  def eigenFaceQuery(v: org.apache.mahout.math.Vector): QueryResponse = {
    val query = new SolrQuery
    query.setRequestHandler("/select")
    val currentPointStr = v.toArray.mkString(",")
    val eigenfaceFieldNames = (0 until v.size()).map(i => s"e${i}_d").mkString(",")
    val distFnStr = s"dist(2, ${eigenfaceFieldNames},${currentPointStr})"
    query.setQuery("*:*")
    query.setSort(new SortClause(distFnStr, SolrQuery.ORDER.asc))
    query.setFields("name_s", "calc_dist:" + distFnStr, "last_seen_pdt")
    query.setRows(10)

    val response: QueryResponse = solrClient.query(query)
    response
  }

  def insertNewFaceToSolr(v: org.apache.mahout.math.Vector): String = {
    // Kept for backwards compatability
    val humanName = "human-" + scala.util.Random.alphanumeric.take(5).mkString("").toUpperCase
    addFaceToSolr(v, humanName)
    commit()
    humanName
  }

  def getDocsArray(response: QueryResponse): Array[SolrDocument] = {
    val a = new Array[SolrDocument](response.getResults.size())
    for (i <- 0 until response.getResults.size()) {
      a(i) = response.getResults.get(i)
    }
    a
  }

  def addFaceToSolr(v: org.apache.mahout.math.Vector, name: String): Unit = {
    val doc = new SolrInputDocument()
    doc.addField("name_s", name)
    doc.addField("last_seen_pdt", ZonedDateTime.now.format(DateTimeFormatter.ISO_INSTANT)) // YYYY-MM-DDThh:mm:ssZ   DateTimeFormatter.ISO_INSTANT, ISO-8601
    v.toMap.map { case (k, v) => doc.addField(s"e${k.toString}_d", v) }
    solrClient.add(doc)
    solrClient.commit()
  }

  def commit(): Unit = {
    solrClient.commit()
  }
}
