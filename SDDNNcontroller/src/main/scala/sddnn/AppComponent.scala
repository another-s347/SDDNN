package sddnn

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
import sddnn.Intent.{Ipv4Constraint, LinkCollectionIntentCompiler, TableIdConstraint}
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
    var packetProcessor = new MyPacketProcessor
    var task: mutable.ArrayBuffer[PortStatsTask] = mutable.ArrayBuffer.empty
    var devices: Set[DeviceId] = Set.empty
    val myLinkWeigher = new MyLinkWeigher
    val static_cn: Array[String] = Array("cn1","cn2","cn3","cn4")
    val static_cn_name = Map("10.0.0.5" -> "cn1", "10.0.0.6" -> "cn2", "10.0.0.7" -> "cn3", "10.0.0.8" -> "cn4", "10.0.0.9"->"ed1", "10.0.0.10"->"ed2")
    val intent_map = new java.util.concurrent.ConcurrentHashMap[(HostId,HostId),(Intent,Intent)]().asScala
    var monitor = new java.util.concurrent.ConcurrentHashMap[(HostId,HostId),MonitorStatus]().asScala
    val use_qos = true
    var link_extra_cost = new ConcurrentHashMap[Link,Double]()
        .asScala
        .withDefaultValue(0.0)

    @Activate def activate(context: ComponentContext): Unit = {
        //cfgService.registerProperties(getClass)
        appId = coreService.registerApplication("sddnn")

        packetService.addProcessor(packetProcessor, PacketProcessor.director(2))
        requestIntercepts()
        intentService.getIntents.forEach(intent => {
            if (intent.appId() == appId) {
                intentService.withdraw(intent)
                intentService.purge(intent)
            }
        })
        log.info("scala Started")
    }

    def requestIntercepts(): Unit = {
        val selector = DefaultTrafficSelector.builder()
        selector.matchEthType(Ethernet.TYPE_IPV4)
        packetService.requestPackets(selector.build(), PacketPriority.REACTIVE, appId)
    }

    @Deactivate def deactivate(): Unit = {
        withdrawIntercepts()
        packetService.removeProcessor(packetProcessor)
        packetProcessor = null
        task.foreach(t=>{
            t.cancel()
        })
        task.clear()
        log.info("Stopped")
    }

    def withdrawIntercepts(): Unit = {
        val selector = DefaultTrafficSelector.builder()
        selector.matchEthType(Ethernet.TYPE_IPV4)
        packetService.cancelPackets(selector.build(), PacketPriority.REACTIVE, appId)
    }

    private def toMacAddress(deviceId: DeviceId) = { // Example of deviceId.toString(): "of:0000000f6002ff6f"
        // The associated MAC address is "00:0f:60:02:ff:6f"
        val tmp1 = deviceId.toString.substring(7)
        val tmp2 = tmp1.substring(0, 2) + ":" + tmp1.substring(2, 4) + ":" + tmp1.substring(4, 6) + ":" + tmp1.substring(6, 8) + ":" + tmp1.substring(8, 10) + ":" + tmp1.substring(10, 12)
        //log.info("toMacAddress: deviceId = {}, Mac = {}", deviceId.toString(), tmp2);
        MacAddress.valueOf(tmp2)
    }

    class MyPacketProcessor extends PacketProcessor {
        override def process(context: PacketContext): Unit = {
            if (context.isHandled) {
                return
            }

            val inPacket: InboundPacket = context.inPacket()
            val ethPacket = inPacket.parsed()

            if (ethPacket == null) return

            if (ethPacket.getEtherType == Ethernet.TYPE_IPV6 && ethPacket.isMulticast) {
                return
            }

            if (ethPacket.getEtherType != Ethernet.TYPE_IPV4) {
                log.info(f"ethernet type == ${ethPacket.getEtherType}, ignore...")
                return
            }
            val ipv4 = ethPacket.getPayload.asInstanceOf[IPv4]
            val ipv4_src = Ip4Address.valueOf(ipv4.getSourceAddress)
            val ipv4_dst = Ip4Address.valueOf(ipv4.getDestinationAddress)

            val srchosts = hostService.getHostsByIp(ipv4_src)
            if(srchosts.size()!=1) {
//                log.info(s"unknown size of host ip == ${ipv4_src}, ${srchosts.size()}")
                flood(context)
                return
            }
            val srcHost:Host = srchosts.iterator().next()
            val src_is_cn = static_cn_name.get(ipv4_src.toString)
            val dsthosts = hostService.getHostsByIp(ipv4_dst)
            if(dsthosts.size()!=1) {
//                log.info(s"unknown size of host ip == ${ipv4_dst}, ${dsthosts.size()}")
                flood(context)
                return
            }
            val dstHost:Host = dsthosts.iterator().next()
            val dst_is_cn = static_cn_name.get(ipv4_dst.toString)

            if(intent_map.contains((srcHost.id(),dstHost.id()))) {
                return;
            }
            if(intent_map.contains((dstHost.id(),srcHost.id()))) {
                return;
            }
            val paths = if(src_is_cn.isDefined&&dst_is_cn.isDefined&&use_qos) {
                pathService.getPaths(srcHost.id(),dstHost.id(), myLinkWeigher)
            }
            else {
                pathService.getPaths(srcHost.id(),dstHost.id())
            }
            if(paths.size()==0) {
//                log.info(s"unable to find path ${srcHost}=>${dstHost}")
                return
            }

            val path:Path = paths.iterator().next()
            val firstLink = path.links().get(0)
            val lastLink = path.links().get(path.links().size() - 1)
            val one_intent_builder = PointToPointIntent.builder().appId(appId)
                .selector(DefaultTrafficSelector.builder()
                    .matchEthType(Ethernet.TYPE_IPV4)
                    .matchIPSrc(IpPrefix.valueOf(ipv4_src, IpPrefix.MAX_INET_MASK_LENGTH))
                    .matchIPDst(IpPrefix.valueOf(ipv4_dst, IpPrefix.MAX_INET_MASK_LENGTH))
                    .build())
                .priority(8000)
                .filteredIngressPoint(new FilteredConnectPoint(firstLink.dst()))
                .filteredEgressPoint(new FilteredConnectPoint(lastLink.src()))
            val one_intent = one_intent_builder
                .suggestedPath(path.links().subList(1, path.links().size() - 1))
                .build()
            intentService.submit(one_intent)

            val two_intent_builder = PointToPointIntent.builder().appId(appId)
                .selector(DefaultTrafficSelector.builder()
                    .matchEthType(Ethernet.TYPE_IPV4)
                    .matchIPSrc(IpPrefix.valueOf(ipv4_dst, IpPrefix.MAX_INET_MASK_LENGTH))
                    .matchIPDst(IpPrefix.valueOf(ipv4_src, IpPrefix.MAX_INET_MASK_LENGTH))
                    .build())
                .priority(8000)
                .filteredIngressPoint(new FilteredConnectPoint(lastLink.src()))
                .filteredEgressPoint(new FilteredConnectPoint(firstLink.dst()))

            val suggestedPath = path
                .links()
                .subList(1, path.links().size() - 1)
                .asScala
                .map(reverse_link)
                .reverse
                .asJava
            val two_intent = two_intent_builder
                .suggestedPath(suggestedPath)
                .build()
            intentService.submit(two_intent)

            log.info(f"submit intent ${srcHost}<=>${dstHost}")

            intent_map.put((srcHost.id(),dstHost.id()),(one_intent,two_intent))

            if(use_qos) {
                if(!(src_is_cn.isDefined&&dst_is_cn.isDefined)) {
                    val t = new PortStatsTask(MonitorStatus(path,one_intent,two_intent,srcHost,dstHost,one_intent_builder,two_intent_builder))
                    t.schedule()
                    log.info(f"add qos reroute task")
                    task.addOne(t)
                }
                else {
                    path.links().forEach(link=>{
                        val r = reverse_link(link)
                        link_extra_cost.put(link, Double.PositiveInfinity)
                        link_extra_cost.put(r, Double.PositiveInfinity)
                    })
                }
            }
        }

        def findNextHop(curDevice: DeviceId, path: Path): Option[Link] = {
            path.links().forEach(link => {
                if (link.src().elementId() == curDevice) {
                    return Some(link)
                }
            })
            None
        }

        def flood(context: PacketContext): Unit = {
            if (topologyService.isBroadcastPoint(topologyService.currentTopology(), context.inPacket().receivedFrom())) {
                packetOut(context, PortNumber.FLOOD)
            }
            else {
//                log.info("block at device == {}", context.inPacket().receivedFrom().deviceId())
//                log.info("block at port == {}", context.inPacket().receivedFrom().port())
                context.block()
            }
        }

        def packetOut(context: PacketContext, connectPoint: ConnectPoint): Unit = {
            val device = connectPoint.deviceId()
            val treatment = DefaultTrafficTreatment.builder().setOutput(connectPoint.port()).build()
            val packet = new DefaultOutboundPacket(device, treatment, context.inPacket().unparsed())
            packetService.emit(packet)
        }

        def packetOut(context: PacketContext, number: PortNumber): Unit = {
            context.treatmentBuilder().setOutput(number)
            context.send()
        }

        def packetOut(context: PacketContext): Unit = {
            context.send()
        }

        def packetOut(ethPacket: IPacket, connectPoint: ConnectPoint):Unit = {
            val device = connectPoint.deviceId()
            val treatment = DefaultTrafficTreatment.builder().setOutput(connectPoint.port()).build()
            val packet = new DefaultOutboundPacket(device, treatment, ByteBuffer.wrap(ethPacket.serialize()))
            packetService.emit(packet)
        }

        def udpPacketOut(connectPoint: ConnectPoint, dstMac:MacAddress, dstIpv4:String, udpSrcPort:Int, udpDstPort:Int, data:Array[Byte]):Unit = {
            val response = (new Ethernet)
                .setDestinationMACAddress(dstMac)
                .setSourceMACAddress("00:00:00:00:00:00")
                .setEtherType(Ethernet.TYPE_IPV4)
                .setPayload((new IPv4)
                    .setSourceAddress("10.0.0.254")
                    .setDestinationAddress(dstIpv4)
                    .setProtocol(17)
                    .setTtl(64)
                    .setPayload((new UDP)
                        .setDestinationPort(udpDstPort)
                        .setSourcePort(udpSrcPort)
                        .setPayload(new Data(data))
                    )
                )
            packetOut(response, connectPoint)
        }
    }

    def reverse_link(link:Link):Link = {
        DefaultLink.builder()
            .`type`(link.`type`())
            .annotations(link.annotations())
            .isExpected(link.isExpected)
            .providerId(link.providerId())
            .state(link.state())
            .src(link.dst())
            .dst(link.src())
            .build().asInstanceOf[Link]
    }

    case class MonitorStatus(path: Path, one_intent: Intent,two_intent: Intent, src: Host, dst: Host, one_intentBuilder: PointToPointIntent.Builder, two_intentBuilder: PointToPointIntent.Builder)

    class MyLinkWeigher extends LinkWeigher {
        override def weight(edge: TopologyEdge): Weight = {
            val link = edge.link()
            link.`type`() match {
                case Link.Type.DIRECT =>
                    val extra = link_extra_cost(link)
                    val egressSrc = portStatService.load(link.src(), MetricType.BYTES)
                    val egressDst = portStatService.load(link.dst(), MetricType.BYTES)
                    val rate = egressDst.rate().max(egressSrc.rate())
                    val rate_kbs = rate*8/(1024).toDouble // Kb/s
                    val cost = if(rate_kbs>300.0) {
                        10000.0
                    }
                    else {
                        val k = 300.0/(300.0-rate_kbs)
                        if(k>10000.0) 10000.0 else k
                    }
                    new ScalarWeight(cost+extra)
                case _ => new ScalarWeight(1)
            }
        }

        override def getInitialWeight: Weight = {
            new ScalarWeight(1)
        }

        override def getNonViableWeight: Weight = {
            new ScalarWeight(1)
        }
    }

    class PortStatsTask(var m: MonitorStatus) {
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
                if (use_qos) {
                    val newPathIter = pathService.getPaths(m.src.id(), m.dst.id(), myLinkWeigher).iterator().asScala.toList
                    if (newPathIter.nonEmpty) {
                        val path = newPathIter.head
                        if (!path.equals(m.path)) {
//                            if(path.weight().asInstanceOf[ScalarWeight].value().isPosInfinity) {
//                                log.info(f"switch...path: ${path}\noptions: ${newPathIter}")
//                            }
//                            else {
//                                log.info(f"switch...path: ${path}")
//                            }
                            val one_builder: PointToPointIntent.Builder = m.one_intentBuilder
                            one_builder.suggestedPath(path.links().subList(1, path.links().size() - 1))
                            val one_intent = one_builder.build()

                            val two_intent_builder = m.two_intentBuilder

                            val suggestedPath = path.links()
                                .subList(1, path.links().size() - 1)
                                .asScala
                                .map(reverse_link)
                                .reverse
                                .asJava
                            val two_intent = two_intent_builder
                                .suggestedPath(suggestedPath)
                                .build()

                            m = MonitorStatus(path, one_intent, two_intent, m.src, m.dst, m.one_intentBuilder, m.two_intentBuilder)
                            intentService.submit(one_intent)
                            intentService.submit(two_intent)
                        }
                    }
                }
            }
        }

    }
}