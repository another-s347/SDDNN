package sddnn

import java.util

import com.google.common.collect.ImmutableList
import org.onlab.packet.{Ethernet, Ip4Address, Ip4Prefix, MacAddress}
import org.onosproject.core.ApplicationId
import org.onosproject.net.flow.{DefaultFlowRule, DefaultTrafficSelector, DefaultTrafficTreatment, FlowRule}
import org.onosproject.net.host.HostService
import org.onosproject.net.intent.{Constraint, FlowRuleIntent, Intent, IntentCompiler, LinkCollectionIntent, PathIntent, ResourceContext}
import org.onosproject.net._
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

object Intent {

    object Ipv4Constraint {
        def of(host: HostId, ip: Ip4Address): Ipv4Constraint = {
            new Ipv4Constraint(host, ip)
        }
    }

    class Ipv4Constraint(val host: HostId, val ip: Ip4Address) extends Constraint {

        override def cost(link: Link, context: ResourceContext): Double = 1

        override def validate(path: Path, context: ResourceContext): Boolean = true
    }

    object TableIdConstraint {
        def of(tableId: Int): TableIdConstraint = {
            new TableIdConstraint(tableId)
        }
    }

    class TableIdConstraint(val tableId: Int) extends Constraint {
        override def validate(path: Path, context: ResourceContext): Boolean = true

        override def cost(link: Link, context: ResourceContext): Double = 1
    }

    class LinkCollectionIntentCompiler(hostService: HostService, appId: ApplicationId) extends IntentCompiler[LinkCollectionIntent] {
        final private val log = LoggerFactory.getLogger(getClass)

        override def compile(intent: LinkCollectionIntent, installable: util.List[Intent]): util.List[Intent] = {
            val rules = ImmutableList.builder[FlowRule]()
            val intents = ImmutableList.builder[Intent]()
            if (intent.ingressPoints().size() != 1) {
                return intents.build()
            }
            if (intent.egressPoints().size() != 1) {
                return intents.build()
            }
            val src = intent.ingressPoints().iterator().next()
            val dst = intent.egressPoints().iterator().next()
            log.info(f"compile intent $src --> $dst")
            val srcHosts = hostService.getConnectedHosts(src)
            if (srcHosts.size() != 1) {
                return intents.build()
            }
            val srcHost = srcHosts.iterator().next()
            val dstHosts = hostService.getConnectedHosts(dst)
            if (dstHosts.size() != 1) {
                return intents.build()
            }
            val dstHost = dstHosts.iterator().next()
            val nodes = scala.collection.mutable.HashMap.empty[DeviceId, Node]
            if(intent.links().size()==0) {
                nodes.put(src.deviceId(),LocalNode(src.deviceId(),src.port(),dst.port()))
            }
            intent.links().forEach(link => {
                log.info(f"link == $link")
                // src* --> ???
                if (src.deviceId() == link.src().deviceId()) {
                    nodes.put(src.deviceId(), SrcNode(src.deviceId(), inPort = src.port(), outPort = link.src().port(), next = link.dst().deviceId()))
                    // src --> dst*
                    if (link.dst().deviceId() == dst.deviceId()) {
                        nodes.put(link.dst().deviceId(), DstNode(dst.deviceId(), inPort = link.dst().port(), outPort = dst.port(), prev = link.src().deviceId()))
                    }
                    else { // src --> inter*
                        nodes.get(link.dst().deviceId()) match {
                            case None =>
                                nodes.put(link.dst().deviceId(), InterNode(link.dst().deviceId(), inPort = link.dst().port(), outPort = PortNumber.ANY, prev = link.src().deviceId(), next = DeviceId.NONE))
                            case Some(inter: InterNode) =>
                                nodes.put(link.dst().deviceId(), InterNode(link.dst().deviceId(), inPort = link.dst().port(), outPort = inter.outPort, prev = link.src().deviceId(), next = inter.next))
                            case _ =>
                        }
                    }
                }
                else {
                    // inter* --> ?
                    nodes.get(link.src().deviceId()) match {
                        case None =>
                            nodes.put(link.src().deviceId(), InterNode(link.src().deviceId(), inPort = PortNumber.ANY, outPort = link.src().port(), prev = DeviceId.NONE, next = link.dst().deviceId()))
                        case Some(inter: InterNode) =>
                            nodes.put(link.src().deviceId(), InterNode(link.src().deviceId(), inPort = inter.inPort, outPort = link.src().port(), prev = inter.prev, next = link.dst().deviceId()))
                        case _ =>
                    }
                    // inter --> dst*
                    if (link.dst().deviceId() == dst.deviceId()) {
                        nodes.put(link.dst().deviceId(), DstNode(link.dst().deviceId(), inPort = link.dst().port(), outPort = dst.port(), prev = link.src().deviceId()))
                    }
                    else {
                        // inter --> inter*
                        nodes.get(link.dst().deviceId()) match {
                            case None =>
                                nodes.put(link.dst().deviceId(), InterNode(link.dst().deviceId(), inPort = link.dst().port(), outPort = PortNumber.ANY, prev = link.src().deviceId(), next = DeviceId.NONE))
                            case Some(inter: InterNode) =>
                                nodes.put(link.dst().deviceId(), InterNode(link.dst().deviceId(), inPort = link.dst().port(), outPort = inter.outPort, prev = link.src().deviceId(), next = inter.next))
                            case _ =>
                        }
                    }
                }
            })
            nodes.values.foreach(node => {
                rules.add(createFlowRuleFromNode(intent, node, srcHost, dstHost))
            })
            val rulesBuild = rules.build()
            if (!rulesBuild.isEmpty)
                intents.add(new FlowRuleIntent(appId, intent.key, rulesBuild, intent.resources, PathIntent.ProtectionType.PRIMARY, null))
            intents.build()
        }

        def createFlowRuleFromNode(intent: LinkCollectionIntent, node: Node, srcHost: Host, dstHost: Host): FlowRule = {
            val ipv4 = intent.constraints().asScala.flatMap {
                case constraint: Ipv4Constraint =>
                    Some(constraint)
                case _ => None
            }
            val tableId = intent.constraints().asScala.flatMap {
                case constraint: TableIdConstraint => Some(constraint.tableId)
                case _ => None
            }.headOption.getOrElse(0)
            //            val isWireless = intent.constraints().asScala.find(_.isInstanceOf[WirelessConstraint]).isDefined
            val srcIp = ipv4.find(_.host == srcHost.id()).map(_.ip)
            val dstIp = ipv4.find(_.host == dstHost.id()).map(_.ip)
            node match {
                case SrcNode(cur, inPort, outPort, next) =>
                    val selector = DefaultTrafficSelector.builder(intent.selector())
                    selector.matchEthSrc(srcHost.mac())
                        .matchEthDst(dstHost.mac())
                        .matchEthType(Ethernet.TYPE_IPV4)
                    if (srcIp.isDefined) {
                        selector.matchIPSrc(Ip4Prefix.valueOf(srcIp.get, Ip4Prefix.MAX_MASK_LENGTH))
                    }
                    if (dstIp.isDefined) {
                        selector.matchIPDst(Ip4Prefix.valueOf(dstIp.get, Ip4Prefix.MAX_MASK_LENGTH))
                    }
                    val treatment = DefaultTrafficTreatment.builder(intent.treatment())
                    treatment.setEthSrc(toMacAddress(cur))
                        .setEthDst(toMacAddress(next))
                        .setOutput(outPort)
                    DefaultFlowRule.builder
                        .forDevice(cur)
                        .forTable(tableId)
                        .withSelector(selector.build())
                        .withTreatment(treatment.build())
                        .withPriority(intent.priority)
                        .fromApp(appId)
                        .makeTemporary(2000)
                        .build()
                case InterNode(cur, inPort, outPort, prev, next) =>
                    val selector = DefaultTrafficSelector.builder(intent.selector())
                    selector.matchEthSrc(toMacAddress(prev))
                        .matchEthDst(toMacAddress(cur))
                        .matchEthType(Ethernet.TYPE_IPV4)
                    if (srcIp.isDefined) {
                        selector.matchIPSrc(Ip4Prefix.valueOf(srcIp.get, Ip4Prefix.MAX_MASK_LENGTH))
                    }
                    if (dstIp.isDefined) {
                        selector.matchIPDst(Ip4Prefix.valueOf(dstIp.get, Ip4Prefix.MAX_MASK_LENGTH))
                    }
                    val treatment = DefaultTrafficTreatment.builder(intent.treatment())
                    treatment.setEthSrc(toMacAddress(cur))
                        .setEthDst(toMacAddress(next))
                    if (outPort == inPort) {
                        treatment.setOutput(PortNumber.IN_PORT)
                    }
                    else {
                        treatment.setOutput(outPort)
                    }
                    DefaultFlowRule.builder
                        .forDevice(cur)
                        .forTable(tableId)
                        .withSelector(selector.build())
                        .withTreatment(treatment.build())
                        .withPriority(intent.priority)
                        .fromApp(appId)
                        .makeTemporary(2000)
                        .build()
                case DstNode(cur, inPort, outPort, prev) =>
                    val selector = DefaultTrafficSelector.builder(intent.selector())
                    selector.matchEthSrc(toMacAddress(prev))
                        .matchEthDst(toMacAddress(cur))
                        .matchEthType(Ethernet.TYPE_IPV4)
                    if (srcIp.isDefined) {
                        selector.matchIPSrc(Ip4Prefix.valueOf(srcIp.get, Ip4Prefix.MAX_MASK_LENGTH))
                    }
                    if (dstIp.isDefined) {
                        selector.matchIPDst(Ip4Prefix.valueOf(dstIp.get, Ip4Prefix.MAX_MASK_LENGTH))
                    }
                    val treatment = DefaultTrafficTreatment.builder(intent.treatment())
                    treatment.setEthSrc(srcHost.mac())
                        .setEthDst(dstHost.mac())
                        .setOutput(outPort)
                    DefaultFlowRule.builder
                        .forDevice(cur)
                        .forTable(tableId)
                        .withSelector(selector.build())
                        .withTreatment(treatment.build())
                        .withPriority(intent.priority)
                        .fromApp(appId)
                        .makeTemporary(2000)
                        .build()
                case LocalNode(cur, inPort, outPort) =>
                    val selector = DefaultTrafficSelector.builder(intent.selector())
                    selector.matchEthSrc(srcHost.mac())
                        .matchEthDst(dstHost.mac())
                        .matchEthType(Ethernet.TYPE_IPV4)
                    if (srcIp.isDefined) {
                        selector.matchIPSrc(Ip4Prefix.valueOf(srcIp.get, Ip4Prefix.MAX_MASK_LENGTH))
                    }
                    if (dstIp.isDefined) {
                        selector.matchIPDst(Ip4Prefix.valueOf(dstIp.get, Ip4Prefix.MAX_MASK_LENGTH))
                    }
                    val treatment = DefaultTrafficTreatment.builder(intent.treatment())
                    treatment.setOutput(outPort)
                    DefaultFlowRule.builder
                        .forDevice(cur)
                        .forTable(tableId)
                        .withSelector(selector.build())
                        .withTreatment(treatment.build())
                        .withPriority(intent.priority)
                        .fromApp(appId)
                        .makeTemporary(2000)
                        .build()
            }
        }

        private def toMacAddress(deviceId: DeviceId) = { // Example of deviceId.toString(): "of:0000000f6002ff6f"
            // The associated MAC address is "00:0f:60:02:ff:6f"
            val tmp1 = deviceId.toString.substring(7)
            val tmp2 = tmp1.substring(0, 2) + ":" + tmp1.substring(2, 4) + ":" + tmp1.substring(4, 6) + ":" + tmp1.substring(6, 8) + ":" + tmp1.substring(8, 10) + ":" + tmp1.substring(10, 12)
            //log.info("toMacAddress: deviceId = {}, Mac = {}", deviceId.toString(), tmp2);
            MacAddress.valueOf(tmp2)
        }
    }

    sealed trait Node

    case class SrcNode(cur: DeviceId, inPort: PortNumber, outPort: PortNumber, next: DeviceId) extends Node

    case class InterNode(cur: DeviceId, inPort: PortNumber, outPort: PortNumber, prev: DeviceId, next: DeviceId) extends Node

    case class DstNode(cur: DeviceId, inPort: PortNumber, outPort: PortNumber, prev: DeviceId) extends Node

    case class LocalNode(cur: DeviceId, inPort: PortNumber, outPort: PortNumber) extends Node

}
