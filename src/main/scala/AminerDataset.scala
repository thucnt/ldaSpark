
import edu.stanford.nlp.process.Morphology
import edu.stanford.nlp.simple.Document
import org.apache.log4j.{Level, Logger}
import scala.collection.JavaConversions._
import scalax.io._

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, OnlineLDAOptimizer}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}


class AminerDataset(sc: SparkContext, spark: SparkSession) {

  def run(params: Params): Unit = {

    Logger.getRootLogger.setLevel(Level.WARN)

    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens) =
      preprocess(sc, params.input, params.vocabSize, params.stopwordFile)
    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.length
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9
    corpus.cache()
    println()
    println(s"Corpus summary:")
    println(s"\t Training set size: $actualCorpusSize documents")
    println(s"\t Vocabulary size: $actualVocabSize terms")
    println(s"\t Training set size: $actualNumTokens tokens")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()
    corpus.coalesce(1).saveAsTextFile("src/main/resources/testresult/corpus")


//    val output:Output = Resource.fromFile("someFile")
//
//    // Note: each write will open a new connection to file and
//    //       each write is executed at the begining of the file,
//    //       so in this case the last write will be the contents of the file.
//    // See Seekable for append and patching files
//    // Also See openOutput for performing several writes with a single connection
//
//    output.writeIntsAsBytes(1,2,3)
//    output.write("hello")(Codec.UTF8)
//    output.writeStrings(List("hello","world")," ")(Codec.UTF8)

    // Run LDA.
    sc.stop()
  }

  import org.apache.spark.sql.functions._


  /**
    * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
    *
    * @return (corpus, vocabulary as array, total token count in corpus)
    */
  def preprocess(
                  sc: SparkContext,
                  paths: String,
                  vocabSize: Int,
                  stopwordFile: String): (RDD[(Long, Vector)], Array[String], Long) = {

    import spark.implicits._
    //Reading the Whole Text Files
    val initialrdd = spark.sparkContext.wholeTextFiles(paths).map(_._2)
//    val initialrdd = spark.sparkContext.textFile(paths)
//        .map(x => {
//          x.substring(x.indexOf(";;;") + 3)
//        })
    initialrdd.cache()
    val rdd = initialrdd.mapPartitions { partition =>
      val morphology = new Morphology()
      partition.map { value =>
        TextHelper.getLemmaText(value, morphology)
      }
    }.map(TextHelper.filterSpecialCharacters)
    rdd.cache()
    initialrdd.unpersist()
    val df = rdd.toDF("docs")
    val customizedStopWords: Array[String] = if (stopwordFile.isEmpty) {
      Array.empty[String]
    } else {
      val stopWordText = sc.textFile(stopwordFile).collect()
      stopWordText.flatMap(_.stripMargin.split(","))
    }
    //Tokenizing using the RegexTokenizer
    val tokenizer = new RegexTokenizer().setInputCol("docs").setOutputCol("rawTokens")

    //Removing the Stop-words using the Stop Words remover
    val stopWordsRemover = new StopWordsRemover().setInputCol("rawTokens").setOutputCol("tokens")
    stopWordsRemover.setStopWords(stopWordsRemover.getStopWords ++ customizedStopWords)

    //Converting the Tokens into the CountVector
    val countVectorizer = new CountVectorizer().setVocabSize(vocabSize).setInputCol("tokens").setOutputCol("features")

    //Setting up the pipeline
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, countVectorizer))

    val model = pipeline.fit(df)
    val documents = model.transform(df).select("features").rdd.map {
      case Row(features: MLVector) => Vectors.fromML(features)
    }.zipWithIndex().map(_.swap)

    (documents,
      model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary, // vocabulary
      documents.map(_._2.numActives).sum().toLong) // total token count
  }
}

object AminerDataset extends App {

  val conf = new SparkConf().setAppName(s"AminerDataset").setMaster("local[*]").set("spark.executor.memory", "8g")
  val spark = SparkSession.builder().config(conf).getOrCreate()
  val sc = spark.sparkContext
  val aminer = new AminerDataset(sc, spark)
  val defaultParams = Params().copy(input = "src/main/resources/docs/")
  aminer.run(defaultParams)
//  val fileRdd = spark.sparkContext.textFile("/Users/thucnt/git/ETD/data/papers.txt")
//    .map(x => {
//      x.substring(x.indexOf(" ") + 1)
//    })
//  fileRdd.cache()
//  val rdd = fileRdd.mapPartitions { partition =>
//    val morphology = new Morphology()
//    partition.map { value =>
//      TextHelper.getLemmaText(value, morphology)
//    }
//  }.map(TextHelper.filterSpecialCharacters)
//  rdd.cache()
  //  val fileRdd = spark.sparkContext.wholeTextFiles("src/main/resources/docs/").map(_._1)

  //  val splitRdd = fileRdd.map( line => line.split(";;;") )// RDD[ Array[ String ]
  //  splitRdd.foreach({arr => {println(arr(1))}})

  //  val yourRdd = splitRdd.flatMap( arr => {
  //    val id = arr( 0 )
  //    val text = arr( 1 )
  //    val words = text.split( " " )
  //    words.map( word => ( word, id ) )
  //  } )// RDD[ ( String, String ) ]

  // Now, if you want to print this...
  //yourRdd.foreach( { case ( word, id ) => println( s"{ $word, $id }" ) } )

}

object TextHelper {

  def filterSpecialCharacters(document: String) = document.replaceAll("""[! @ # $ % ^ & * ( ) _ + - − , " ' ; : . ` ? --]""", " ")

  def getStemmedText(document: String) = {
    val morphology = new Morphology()
    new Document(document).sentences().toList.flatMap(_.words().toList.map(morphology.stem)).mkString(" ")
  }

  def getLemmaText(document: String, morphology: Morphology) = {
    val string = new StringBuilder()
    val value = new Document(document).sentences().toList.flatMap { a =>
      val words = a.words().toList
      val tags = a.posTags().toList
      (words zip tags).toMap.map { a =>
        val newWord = morphology.lemma(a._1, a._2)
        val addedWoed = if (newWord.length > 3) {
          newWord
        } else {
          ""
        }
        string.append(addedWoed + " ")
      }
    }
    string.toString()
  }
}