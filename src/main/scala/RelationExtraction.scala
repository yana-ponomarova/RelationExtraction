import java.util.Properties
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.semgraph.SemanticGraph
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.JavaConversions._
import edu.stanford.nlp.pipeline.Annotation



/**
 * Created by yana on 07/01/2016.
 */
object RelationExtraction {


  def main(args: Array[String]) {



    //val inputDir = "src/main/resources/sentences.txt"
    val inputDir = "src/main/resources/oil_supplychain.txt"
    val conf = new SparkConf().setAppName("Relation Extraction").setMaster("local")
    val sc = new SparkContext(conf)
    val textFile = sc.textFile(inputDir).map(line => line.split("\\|")(1))

    val properties = new Properties()
    // annotator parse needs ssplit and tokenize
    properties.setProperty("annotators", "tokenize, ssplit, parse, lemma")
    //properties.setProperty("annotators", "tokenize, ssplit, parse, lemma, natlog, openie")

    //val s = "Gaz de France provides the site with its fuel gas requirements."
    // val s = "Gaz de France provides around 30-40% of the fuel gas requirements for the site."
    //val s = "Crude Oil is available from the Kumkol Oil Field."
    //val s = "The refinery processes local Hassi Messaoud Crude Oil, which is supplied by pipeline."
    //val s = "Obama was born in Hawaii. He is our president."
    //println(getRelationships(s,  new StanfordCoreNLP(properties)))


    //val fw = new FileWriter("target/test.txt", true)

    //textFile.map(line => getRelationships(line, new StanfordCoreNLP(properties))).foreach(println)
    //textFile.map(line => getRelationships(line, new StanfordCoreNLP(properties), new FileWriter("test_2.txt", true)))

    textFile.map(line => getRelationships(line, new StanfordCoreNLP(properties))).saveAsTextFile("/home/osboxes/Documents/Dev/RelationExtraction - dev/output/test3.txt")
    sc.stop()
  }

  def relationParsing(semGraph: SemanticGraph): (String, String, String, String) = {

    import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations

    /*

    Determine the voice of the sentence :
    If the sentence is in active voice, a 'nsubj' dependency should exist.
    If the sentence is in passive voice a 'nsubjpass' dependency should exist

    */
    var voice : String = ""

    if (semGraph.findAllRelns(UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT).length > 0) {
      voice = "active"
    }
    if (semGraph.findAllRelns(UniversalEnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT).length > 0) {
      voice = "passive"
    }

    println(voice)
    var res : (String, String, String, String) = ("","","","")
    val root = semGraph.getFirstRoot()

    val set_modifiers = Set(UniversalEnglishGrammaticalRelations.ADJECTIVAL_MODIFIER, UniversalEnglishGrammaticalRelations.COMPOUND_MODIFIER, UniversalEnglishGrammaticalRelations.NUMERIC_MODIFIER, UniversalEnglishGrammaticalRelations.getNmod("of"))
    val location_modifiers_set = Set(UniversalEnglishGrammaticalRelations.getNmod("in"), UniversalEnglishGrammaticalRelations.getNmod("at"), UniversalEnglishGrammaticalRelations.getNmod("from"))

    val set_outgoing_verbs = Set("pipe", "supply", "export", "send", "provide", "render", "distribute", "sell", "ply", "deliver", "transport", "transfer", "transmit", "channel", "send")
    val incoming_verbs_set = Set("receive", "get", "obtain", "incur ", "acquire", "buy", "purchase", "charter", "take", "bring", "source", "gather", "collect", "import", "extract", "derive", "procure")
    val ambiguous_verbs_set = Set ("ship")
    val arrival_verbs_set = Set("come", "arrive", "get")
    val usage_verbs_set = Set ("use", "consume", "enjoy", "benefit", "employ", "apply", "exploit", "tap", "utilize", "")

    var supplier_full_string : String = ""
    var receiver_full_string : String = ""
    var theme_full_string : String = ""

    val copula = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.COPULA)
    if (copula != null) {
      val children = semGraph.childRelns(root)
      val theme = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT)
      val supplier =  semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.getNmod("from"))
      if (theme != null) {
        val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
        val theme_full = theme_modifiers + theme //++ theme_nmod_location ++ theme_nmod_location_modifiers
        val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
        theme_full_string = theme_full_sorted.map(f => f.word()).mkString(" ")
      }

      if (supplier != null) {
        val supplier_modifiers = semGraph.getChildrenWithRelns(supplier, set_modifiers)
        val supplier_full = supplier_modifiers + supplier
        val supplier_full_sorted = supplier_full.toList.sortWith(_.index() < _.index())
        supplier_full_string = supplier_full_sorted.map(f=> f.word()).mkString(" ")
      }

      res = (root.lemma(), supplier_full_string, receiver_full_string, theme_full_string)

    }

    if (voice == "active") {

      if (set_outgoing_verbs.contains(root.lemma())) {

        val children = semGraph.childRelns(root)
        val supplier = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT)
        val obj =  semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.DIRECT_OBJECT)
        if (obj != null) {
          val obj_nmod =  semGraph.getChildWithReln(obj, UniversalEnglishGrammaticalRelations.getNmod("with"))

          if (obj_nmod == null) {
            val theme = obj
            val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
            val theme_full = theme_modifiers + theme //++ theme_nmod_location ++ theme_nmod_location_modifiers
            val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
            theme_full_string = theme_full_sorted.map(f => f.word()).mkString(" ")

            val receiver_modifiers_set = Set(UniversalEnglishGrammaticalRelations.getNmod("to"), UniversalEnglishGrammaticalRelations.getNmod("for"))
            val theme_receivers = semGraph.getChildrenWithRelns(theme, receiver_modifiers_set)
            if (theme_receivers.size() > 0) {
              val theme_receivers_modifiers = theme_receivers.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers))
              val receivers_full = theme_receivers ++ theme_receivers_modifiers
              val receivers_full_sorted = receivers_full.toList.sortWith(_.index() < _.index())
              receiver_full_string = receivers_full_sorted.map(f => f.word()).mkString(" ")
            }
          }

          if (obj_nmod != null) {
            val theme_receiver = obj
            val theme_receiver_modifiers = semGraph.getChildrenWithRelns(theme_receiver, set_modifiers)
            val receiver_full = theme_receiver_modifiers +theme_receiver
            val receiver_full_sorted = receiver_full.toList.sortWith(_.index() < _.index())
            receiver_full_string = receiver_full_sorted.map(f => f.word()).mkString(" ")

            val theme = obj_nmod
            val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
            val theme_full = theme_modifiers + theme //++ theme_nmod_location ++ theme_nmod_location_modifiers
            val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
            theme_full_string = theme_full_sorted.map(f => f.word()).mkString(" ")

          }

        }

        if (supplier != null) {
          val supplier_modifiers = semGraph.getChildrenWithRelns(supplier, set_modifiers)
          val supplier_full = supplier_modifiers + supplier
          val supplier_full_sorted = supplier_full.toList.sortWith(_.index() < _.index())
          supplier_full_string = supplier_full_sorted.map(f=> f.word()).mkString(" ")
        }

        res = (root.lemma(), supplier_full_string, receiver_full_string, theme_full_string)
      }

      if ((incoming_verbs_set.contains(root.lemma())) | (usage_verbs_set.contains(root.lemma()))) {

        val children = semGraph.childRelns(root)
        val receiver = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT)
        val theme =  semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.DIRECT_OBJECT)
        val supplier =  semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.getNmod("from"))

        if (receiver != null) {
          val receiver_modifiers = semGraph.getChildrenWithRelns(receiver, set_modifiers)
          val receiver_full = receiver_modifiers + receiver
          val receiver_full_sorted = receiver_full.toList.sortWith(_.index() < _.index())
          receiver_full_string = receiver_full_sorted.map(f=> f.word()).mkString(" ")
        }

        if (theme != null) {
          val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
          val theme_full = theme_modifiers + theme //++ theme_nmod_location ++ theme_nmod_location_modifiers
          val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
          theme_full_string = theme_full_sorted.map(f => f.word()).mkString(" ")
        }

        if (supplier != null) {
          val supplier_modifiers = semGraph.getChildrenWithRelns(supplier, set_modifiers)
          val supplier_nmod_location = semGraph.getChildrenWithRelns(supplier, location_modifiers_set)
          val supplier_nmod_location_modifiers = supplier_nmod_location.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers + UniversalEnglishGrammaticalRelations.CASE_MARKER))

          val supplier_full = supplier_modifiers + supplier ++ supplier_nmod_location ++ supplier_nmod_location_modifiers
          val supplier_full_sorted = supplier_full.toList.sortWith(_.index() < _.index())
          supplier_full_string = supplier_full_sorted.map(f => f.word()).mkString(" ")
        }

        res = (root.lemma(), supplier_full_string, receiver_full_string, theme_full_string)
      }



      if (arrival_verbs_set.contains(root.lemma())) {
        val children = semGraph.childRelns(root)
        val theme = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.NOMINAL_SUBJECT)
        val supplier =  semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.getNmod("from"))

        if (theme != null) {
          val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
          val theme_full = theme_modifiers + theme
          val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
          theme_full_string = theme_full_sorted.map(f=> f.word()).mkString(" ")
        }

        if (supplier != null) {
          val supplier_modifiers = semGraph.getChildrenWithRelns(supplier, set_modifiers)
          val supplier_nmod_location = semGraph.getChildrenWithRelns(supplier, location_modifiers_set)
          val supplier_nmod_location_modifiers = supplier_nmod_location.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers + UniversalEnglishGrammaticalRelations.CASE_MARKER))

          val supplier_full = supplier_modifiers + supplier ++ supplier_nmod_location ++ supplier_nmod_location_modifiers
          val supplier_full_sorted = supplier_full.toList.sortWith(_.index() < _.index())
          supplier_full_string = supplier_full_sorted.map(f => f.word()).mkString(" ")

        }

        res = (root.lemma(), supplier_full_string, "", theme_full_string)
      }
    }
    if (voice == "passive") {
      if (set_outgoing_verbs.contains(root.lemma())) {
        val root = semGraph.getFirstRoot()
        val children = semGraph.childRelns(root)
        val theme = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT)
        val supplier = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.getNmod("agent"))
        val receiver = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.getNmod("to"))

        if (theme != null) {
          val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
          val theme_full = theme_modifiers + theme
          val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
          theme_full_string = theme_full_sorted.map(f => f.word()).mkString(" ")
        }

        if (supplier != null) {
          val supplier_modifiers = semGraph.getChildrenWithRelns(supplier, set_modifiers)
          //val theme_nmod_location = semGraph.getChildrenWithRelns(theme, location_modifiers_set)
          //val theme_nmod_location_modifiers = theme_nmod_location.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers + UniversalEnglishGrammaticalRelations.CASE_MARKER))

          val supplier_full = supplier_modifiers + supplier //++ theme_nmod_location ++ theme_nmod_location_modifiers
          val supplier_full_sorted = supplier_full.toList.sortWith(_.index() < _.index())
          supplier_full_string = supplier_full_sorted.map(f => f.word()).mkString(" ")
        }

        if (receiver !=null) {
          val receiver_modifiers = semGraph.getChildrenWithRelns(receiver, set_modifiers)
          //val theme_nmod_location = semGraph.getChildrenWithRelns(theme, location_modifiers_set)
          //val theme_nmod_location_modifiers = theme_nmod_location.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers + UniversalEnglishGrammaticalRelations.CASE_MARKER))

          val receiver_full = receiver_modifiers + receiver //++ theme_nmod_location ++ theme_nmod_location_modifiers
          val receiver_full_sorted = receiver_full.toList.sortWith(_.index() < _.index())
          receiver_full_string = receiver_full_sorted.map(f => f.word()).mkString(" ")
        }


        res = (root.lemma(), supplier_full_string, receiver_full_string, theme_full_string)
      }
      if (incoming_verbs_set.contains(root.lemma())) {
        val root = semGraph.getFirstRoot()
        val children = semGraph.childRelns(root)
        val theme = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT)
        val supplier = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.getNmod("from"))
        val receiver = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.getNmod("agent"))

        if (theme != null) {
          val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
          val theme_full = theme_modifiers + theme
          val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
          theme_full_string = theme_full_sorted.map(f => f.word()).mkString(" ")
        }

        if (supplier != null) {
          val supplier_modifiers = semGraph.getChildrenWithRelns(supplier, set_modifiers)
          //val theme_nmod_location = semGraph.getChildrenWithRelns(theme, location_modifiers_set)
          //val theme_nmod_location_modifiers = theme_nmod_location.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers + UniversalEnglishGrammaticalRelations.CASE_MARKER))

          val supplier_full = supplier_modifiers + supplier //++ theme_nmod_location ++ theme_nmod_location_modifiers
          val supplier_full_sorted = supplier_full.toList.sortWith(_.index() < _.index())
          supplier_full_string = supplier_full_sorted.map(f => f.word()).mkString(" ")
        }

        if (receiver !=null) {
          val receiver_modifiers = semGraph.getChildrenWithRelns(receiver, set_modifiers)
          //val theme_nmod_location = semGraph.getChildrenWithRelns(theme, location_modifiers_set)
          //val theme_nmod_location_modifiers = theme_nmod_location.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers + UniversalEnglishGrammaticalRelations.CASE_MARKER))

          val receiver_full = receiver_modifiers + receiver //++ theme_nmod_location ++ theme_nmod_location_modifiers
          val receiver_full_sorted = receiver_full.toList.sortWith(_.index() < _.index())
          receiver_full_string = receiver_full_sorted.map(f => f.word()).mkString(" ")
        }


        res = (root.lemma(), supplier_full_string, receiver_full_string, theme_full_string)
      }
    }

    if (voice == "") {
      if (ambiguous_verbs_set.contains(root.lemma())) {
        import edu.stanford.nlp.trees.GrammaticalRelation
        val children = semGraph.childRelns(root)
        val nom_subj = semGraph.getChildWithReln(root, UniversalEnglishGrammaticalRelations.COMPOUND_MODIFIER)
        val theme = semGraph.getChildWithReln(root, GrammaticalRelation.DEPENDENT)
        val theme_from =  semGraph.getChildWithReln(theme, UniversalEnglishGrammaticalRelations.getNmod("from"))
        val theme_to =  semGraph.getChildWithReln(theme, UniversalEnglishGrammaticalRelations.getNmod("to"))

        if (theme != null) {
          if (theme_from != null) {
            val supplier = theme_from
            val receiver = nom_subj

            val supplier_modifiers = semGraph.getChildrenWithRelns(supplier, set_modifiers)
            val supplier_nmod_location = semGraph.getChildrenWithRelns(supplier, location_modifiers_set)
            val supplier_nmod_location_modifiers = supplier_nmod_location.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers + UniversalEnglishGrammaticalRelations.CASE_MARKER))

            val supplier_full = supplier_modifiers + supplier ++ supplier_nmod_location ++ supplier_nmod_location_modifiers
            val supplier_full_sorted = supplier_full.toList.sortWith(_.index() < _.index())
            supplier_full_string = supplier_full_sorted.map(f => f.word()).mkString(" ")

            val receiver_modifiers = semGraph.getChildrenWithRelns(receiver, set_modifiers)
            val receiver_full = receiver_modifiers + receiver
            val receiver_full_sorted = receiver_full.toList.sortWith(_.index() < _.index())
            receiver_full_string = receiver_full_sorted.map(f=> f.word()).mkString(" ")

            val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
            val theme_full = theme_modifiers + theme //++ theme_nmod_location ++ theme_nmod_location_modifiers
            val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
            theme_full_string = theme_full_sorted.map(f => f.word()).mkString(" ")

          }
          else if (theme_from != null) {
            val supplier = nom_subj
            val receiver = theme_to

            val supplier_modifiers = semGraph.getChildrenWithRelns(supplier, set_modifiers)
            val supplier_nmod_location = semGraph.getChildrenWithRelns(supplier, location_modifiers_set)
            val supplier_nmod_location_modifiers = supplier_nmod_location.flatMap(f => semGraph.getChildrenWithRelns(f, set_modifiers + UniversalEnglishGrammaticalRelations.CASE_MARKER))

            val supplier_full = supplier_modifiers + supplier ++ supplier_nmod_location ++ supplier_nmod_location_modifiers
            val supplier_full_sorted = supplier_full.toList.sortWith(_.index() < _.index())
            supplier_full_string = supplier_full_sorted.map(f => f.word()).mkString(" ")

            val receiver_modifiers = semGraph.getChildrenWithRelns(receiver, set_modifiers)
            val receiver_full = receiver_modifiers + receiver
            val receiver_full_sorted = receiver_full.toList.sortWith(_.index() < _.index())
            receiver_full_string = receiver_full_sorted.map(f=> f.word()).mkString(" ")

            val theme_modifiers = semGraph.getChildrenWithRelns(theme, set_modifiers)
            val theme_full = theme_modifiers + theme //++ theme_nmod_location ++ theme_nmod_location_modifiers
            val theme_full_sorted = theme_full.toList.sortWith(_.index() < _.index())
            theme_full_string = theme_full_sorted.map(f => f.word()).mkString(" ")
          }
        }

        res = (root.lemma(), supplier_full_string, receiver_full_string, theme_full_string)
      }

    }

    res
  }

  def getRelationships(s: String, pipeline: StanfordCoreNLP) = {

    val document = new Annotation(s)
    pipeline.annotate(document)
    val sentences = document.get(classOf[SentencesAnnotation])
    var CollapsedSentenceDep = List[SemanticGraph]()
    sentences.foreach(CollapsedSentenceDep ::= _.get(classOf[CollapsedCCProcessedDependenciesAnnotation]))
    var Relations = List[(String, String, String, String)]()
    Relations = CollapsedSentenceDep.map(sg =>relationParsing(sg))
    val CollapsedSentenceDepString = CollapsedSentenceDep.toList.map(sg => sg.toString)
    val pair = sentences zip CollapsedSentenceDepString zip Relations

    //var SentenceTriples = List[Array[RelationTriple]]()
    //sentences.foreach(SentenceTriples ::= _.get(classOf[RelationTriplesAnnotation]))
    //val SentenceTriples = sentences.get(0).get(classOf[RelationTriplesAnnotation])

    /*
  Collection<RelationTriple> triples = sentence.get(NaturalLogicAnnotations.RelationTriplesAnnotation.class);
  // Print the triples
  for (RelationTriple triple : triples) {
    System.out.println(triple.confidence + "\t" +
      triple.subjectLemmaGloss() + "\t" +
      triple.relationLemmaGloss() + "\t" +
      triple.objectLemmaGloss());
  }

  */

    pair.mkString("=>")
  }

  /**
   * method to initialize the Stanford coreNLP pipeline for each partitions
   * @param iter
   * @return a new Tree list RDD
   */

}
