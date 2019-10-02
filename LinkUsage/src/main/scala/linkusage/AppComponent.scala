package linkusage

import java.nio.ByteBuffer
import java.time.temporal.TemporalUnit
import java.time.{Duration, LocalDateTime}
import java.util
import java.util.concurrent.ConcurrentHashMap
import java.util.{Optional, Timer, TimerTask}

import org.apache.felix.scr.annotations.{Component, Reference, ReferenceCardinality}
import org.onlab.graph.{ScalarWeight, Weight}
import org.onlab.packet.{UDP, _}
import org.onlab.util.DataRateUnit
import org.onosproject.cfg.ComponentConfigService
import org.onosproject.core.{ApplicationId, CoreService}
import org.onosproject.net._
import org.onosproject.net.device.DeviceService
import org.onosproject.net.flow._
import org.onosproject.net.flowobjective.FlowObjectiveService
import org.onosproject.net.host.HostService
import org.onosproject.net.intent.constraint.BandwidthConstraint
import org.onosproject.net.intent.{Constraint, HostToHostIntent, Intent, IntentCompiler, IntentExtensionService, IntentService, IntentStore, Key, LinkCollectionIntent, PointToPointIntent}
import org.onosproject.net.link.{LinkProviderRegistry, LinkService, LinkStore}
import org.onosproject.net.packet._
import org.onosproject.net.statistic.{FlowStatisticService, PortStatisticsService, StatisticService}
import org.onosproject.net.topology.{LinkWeigher, PathService, TopologyEdge, TopologyService}
import org.osgi.service.component.ComponentContext
import org.osgi.service.component.annotations.{Activate, Deactivate}
import org.slf4j.LoggerFactory
import linkusage.Intent.{Ipv4Constraint, LinkCollectionIntentCompiler, TableIdConstraint}
import org.onosproject.net.statistic.PortStatisticsService.MetricType
import play.api.libs.json.Json

import scala.collection.JavaConverters._
import scala.collection.mutable

@Component(immediate = true)
class AppComponent {
    final private val log = LoggerFactory.getLogger(getClass)
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var topologyService: TopologyService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var cfgService: ComponentConfigService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var coreService: CoreService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var flowRuleService: FlowRuleService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var packetService: PacketService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var hostService: HostService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var flowObjectiveService: FlowObjectiveService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var deviceService: DeviceService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var pathService: PathService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var statService: StatisticService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var portStatService: PortStatisticsService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var flowStatService: FlowStatisticService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var intentExtension: IntentExtensionService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var intentService: IntentService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var linkService: LinkService = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var linkStore: LinkStore = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var linkProviderRegistry: LinkProviderRegistry = _
    @Reference(cardinality = ReferenceCardinality.MANDATORY_UNARY) var intentStore: IntentStore = _
    var appId: ApplicationId = _
    var throughputTask: Option[ThroughputTask] = None
    val static_cn: Array[String] = Array("cn1","cn2","cn3","cn4")
    val static_cn_name = Map("10.0.0.5" -> "cn1", "10.0.0.6" -> "cn2", "10.0.0.7" -> "cn3", "10.0.0.8" -> "cn4", "10.0.0.9"->"ed1", "10.0.0.10"->"ed2")

    @Activate def activate(context: ComponentContext): Unit = {
        appId = coreService.registerApplication("linkusage")
        throughputTask = Some(new ThroughputTask)
        throughputTask.get.schedule()
        log.info("Link usage perf started")
    }


    @Deactivate def deactivate(): Unit = {
        throughputTask.get.cancel()
        throughputTask = None
        log.info("Stopped")
    }

    class ThroughputTask {
        private var exit: Boolean = false
        val timer = new Timer()

        def schedule(): Unit = {
            timer.scheduleAtFixedRate(new Task(), 0, 2000)
        }

        def isExit: Boolean = {
            exit
        }

        def cancel(): Unit = {
            exit = true
            timer.cancel()
        }

        class Task extends TimerTask {
            override def run(): Unit = {
                var t:Long = 0
                linkService.getActiveLinks.forEach(link=>{
                    if(link.src.elementId().isInstanceOf[HostId] || link.dst.elementId().isInstanceOf[HostId]) {

                    }else{
                        val p = portStatService.load(link.dst(), MetricType.BYTES)
                        if(p == null) {
                            return
                        }
                        t += p.rate()
                    }
                })
                log.info(f"Link usage: ${t*8/1024/2} Kbps")
            }
        }
    }
}