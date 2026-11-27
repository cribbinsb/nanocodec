//! RTP capture files: classic pcap (what Wireshark, tcpdump and `GStreamer`'s
//! `pcapparse` read) and RFC 4571 framing (each packet behind a 16-bit
//! length; what `GStreamer`'s `rtpstreampay` / `rtpstreamdepay` write and
//! read).
//!
//! Written pcap files hold Ethernet/IPv4/UDP frames (`LINKTYPE_ETHERNET`) from
//! 10.0.0.1 to 10.0.0.2, port 5004, timestamped from the RTP clock. Reading
//! takes Ethernet (VLAN tags too), Linux cooked v1/v2, raw IP and BSD
//! loopback captures over IPv4 or IPv6, and returns every UDP payload.

use std::io::{Read, Write};

use crate::error::{Result, RtpError};

/// pcap global header magic numbers (microsecond and nanosecond times).
const PCAP_MAGIC_MICROS: u32 = 0xA1B2_C3D4;
const PCAP_MAGIC_NANOS: u32 = 0xA1B2_3C4D;
const PCAP_VERSION: (u16, u16) = (2, 4);
const PCAP_GLOBAL_HEADER_BYTES: usize = 24;
const PCAP_RECORD_HEADER_BYTES: usize = 16;
const SNAP_LENGTH: u32 = 65_535;
/// Link types (tcpdump.org/linktypes.html).
const LINKTYPE_NULL: u32 = 0;
const LINKTYPE_ETHERNET: u32 = 1;
const LINKTYPE_RAW: u32 = 101;
const LINKTYPE_RAW_OLD: u32 = 12;
const LINKTYPE_LINUX_SLL: u32 = 113;
const LINKTYPE_LINUX_SLL2: u32 = 276;
const ETHERNET_HEADER_BYTES: usize = 14;
const SLL_HEADER_BYTES: usize = 16;
const SLL2_HEADER_BYTES: usize = 20;
const NULL_HEADER_BYTES: usize = 4;
const VLAN_TAG_BYTES: usize = 4;
const ETHERTYPE_IPV4: u16 = 0x0800;
const ETHERTYPE_IPV6: u16 = 0x86DD;
const ETHERTYPE_VLAN: u16 = 0x8100;
const IPV4_HEADER_BYTES: usize = 20;
const IPV6_HEADER_BYTES: usize = 40;
const UDP_HEADER_BYTES: usize = 8;
const IP_PROTOCOL_UDP: u8 = 17;
const IPV4_TTL: u8 = 64;
const IPV4_DONT_FRAGMENT: u16 = 0x4000;
const SOURCE_ADDRESS: [u8; 4] = [10, 0, 0, 1];
const DESTINATION_ADDRESS: [u8; 4] = [10, 0, 0, 2];
const SOURCE_MAC: [u8; 6] = [0x02, 0, 0, 0, 0, 1];
const DESTINATION_MAC: [u8; 6] = [0x02, 0, 0, 0, 0, 2];
/// The RTP port of written captures (RFC 3551 §8's default).
pub const RTP_PORT: u16 = 5004;
/// RFC 4571 length prefix.
const FRAMING_LENGTH_BYTES: usize = 2;

/// One captured datagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedPacket {
    /// Capture time in microseconds.
    pub time_us: u64,
    /// UDP destination port (0 for RFC 4571 streams).
    pub port: u16,
    /// The UDP payload: an RTP or RTCP packet.
    pub data: Vec<u8>,
}

/// True for an RTCP packet sharing the port (RFC 5761 §4: payload types
/// 64-95 with the marker bit are RTCP's 192-223).
#[must_use]
pub fn is_rtcp(data: &[u8]) -> bool {
    data.get(1).is_some_and(|&b| (192..=223).contains(&b))
}

/// Writes a pcap capture.
#[derive(Debug)]
pub struct PcapWriter<W: Write> {
    out: W,
    ip_id: u16,
}

impl<W: Write> PcapWriter<W> {
    /// Start a capture: writes the global header.
    ///
    /// # Errors
    /// The writer fails.
    pub fn new(mut out: W) -> Result<Self> {
        let mut header = Vec::with_capacity(PCAP_GLOBAL_HEADER_BYTES);
        header.extend_from_slice(&PCAP_MAGIC_MICROS.to_le_bytes());
        header.extend_from_slice(&PCAP_VERSION.0.to_le_bytes());
        header.extend_from_slice(&PCAP_VERSION.1.to_le_bytes());
        header.extend_from_slice(&0i32.to_le_bytes()); // thiszone
        header.extend_from_slice(&0u32.to_le_bytes()); // sigfigs
        header.extend_from_slice(&SNAP_LENGTH.to_le_bytes());
        header.extend_from_slice(&LINKTYPE_ETHERNET.to_le_bytes());
        out.write_all(&header)?;
        Ok(Self { out, ip_id: 0 })
    }

    /// Append one RTP packet sent at `time_us`.
    ///
    /// # Errors
    /// A packet too large for one UDP datagram, or the writer fails.
    pub fn write(&mut self, time_us: u64, rtp: &[u8]) -> Result<()> {
        let udp_len = UDP_HEADER_BYTES + rtp.len();
        let ip_len = IPV4_HEADER_BYTES + udp_len;
        let ip_len16 =
            u16::try_from(ip_len).map_err(|_| RtpError::Config("packet over 64 KiB".into()))?;
        let mut frame = Vec::with_capacity(ETHERNET_HEADER_BYTES + ip_len);
        frame.extend_from_slice(&DESTINATION_MAC);
        frame.extend_from_slice(&SOURCE_MAC);
        frame.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
        let ip_start = frame.len();
        frame.push(0x45); // version 4, 5-word header
        frame.push(0);
        frame.extend_from_slice(&ip_len16.to_be_bytes());
        frame.extend_from_slice(&self.ip_id.to_be_bytes());
        frame.extend_from_slice(&IPV4_DONT_FRAGMENT.to_be_bytes());
        frame.push(IPV4_TTL);
        frame.push(IP_PROTOCOL_UDP);
        frame.extend_from_slice(&[0, 0]); // checksum, below
        frame.extend_from_slice(&SOURCE_ADDRESS);
        frame.extend_from_slice(&DESTINATION_ADDRESS);
        let checksum = ipv4_checksum(&frame[ip_start..]);
        frame[ip_start + 10..ip_start + 12].copy_from_slice(&checksum.to_be_bytes());
        frame.extend_from_slice(&RTP_PORT.to_be_bytes());
        frame.extend_from_slice(&RTP_PORT.to_be_bytes());
        frame.extend_from_slice(&(udp_len as u16).to_be_bytes());
        frame.extend_from_slice(&[0, 0]); // UDP checksum optional over IPv4
        frame.extend_from_slice(rtp);
        self.ip_id = self.ip_id.wrapping_add(1);
        let mut record = Vec::with_capacity(PCAP_RECORD_HEADER_BYTES);
        record.extend_from_slice(&((time_us / 1_000_000) as u32).to_le_bytes());
        record.extend_from_slice(&((time_us % 1_000_000) as u32).to_le_bytes());
        record.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        record.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        self.out.write_all(&record)?;
        self.out.write_all(&frame)?;
        Ok(())
    }

    /// The underlying writer.
    pub fn into_inner(self) -> W {
        self.out
    }
}

/// RFC 791 header checksum: one's complement of the one's complement sum.
fn ipv4_checksum(header: &[u8]) -> u16 {
    let mut sum: u32 = header
        .chunks(2)
        .map(|pair| u32::from(u16::from_be_bytes([pair[0], *pair.get(1).unwrap_or(&0)])))
        .sum();
    while sum > 0xFFFF {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

/// Read every UDP payload of a pcap capture.
///
/// # Errors
/// Not a pcap file, an unsupported link type, or a truncated record.
pub fn read_pcap<R: Read>(mut input: R) -> Result<Vec<CapturedPacket>> {
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes)?;
    let header = bytes
        .get(..PCAP_GLOBAL_HEADER_BYTES)
        .ok_or(RtpError::Truncated("pcap header"))?;
    let magic_le = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let magic_be = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let (little, nanos) = match (magic_le, magic_be) {
        (PCAP_MAGIC_MICROS, _) => (true, false),
        (PCAP_MAGIC_NANOS, _) => (true, true),
        (_, PCAP_MAGIC_MICROS) => (false, false),
        (_, PCAP_MAGIC_NANOS) => (false, true),
        _ => return Err(RtpError::Malformed("not a pcap file (pcapng is not read)")),
    };
    let word = |b: &[u8]| {
        let array = [b[0], b[1], b[2], b[3]];
        if little {
            u32::from_le_bytes(array)
        } else {
            u32::from_be_bytes(array)
        }
    };
    let link = word(&header[20..24]);
    let mut packets = Vec::new();
    let mut at = PCAP_GLOBAL_HEADER_BYTES;
    while at < bytes.len() {
        let record = bytes
            .get(at..at + PCAP_RECORD_HEADER_BYTES)
            .ok_or(RtpError::Truncated("pcap record header"))?;
        let seconds = u64::from(word(&record[0..4]));
        let fraction = u64::from(word(&record[4..8]));
        let length = word(&record[8..12]) as usize;
        let start = at + PCAP_RECORD_HEADER_BYTES;
        let frame = bytes
            .get(start..start + length)
            .ok_or(RtpError::Truncated("pcap record"))?;
        at = start + length;
        let time_us = seconds * 1_000_000 + if nanos { fraction / 1000 } else { fraction };
        if let Some((port, data)) = udp_payload(link, frame) {
            packets.push(CapturedPacket {
                time_us,
                port,
                data: data.to_vec(),
            });
        }
    }
    Ok(packets)
}

/// The UDP destination port and payload of one captured frame.
fn udp_payload(link: u32, frame: &[u8]) -> Option<(u16, &[u8])> {
    let be16 = |b: &[u8], at: usize| Some(u16::from_be_bytes([*b.get(at)?, *b.get(at + 1)?]));
    let (ethertype, ip) = match link {
        LINKTYPE_ETHERNET => {
            let mut at = ETHERNET_HEADER_BYTES - 2;
            let mut ethertype = be16(frame, at)?;
            while ethertype == ETHERTYPE_VLAN {
                at += VLAN_TAG_BYTES;
                ethertype = be16(frame, at)?;
            }
            (ethertype, frame.get(at + 2..)?)
        }
        LINKTYPE_LINUX_SLL => (
            be16(frame, SLL_HEADER_BYTES - 2)?,
            frame.get(SLL_HEADER_BYTES..)?,
        ),
        LINKTYPE_LINUX_SLL2 => (be16(frame, 0)?, frame.get(SLL2_HEADER_BYTES..)?),
        LINKTYPE_RAW | LINKTYPE_RAW_OLD => {
            let version = frame.first()? >> 4;
            let ethertype = if version == 6 {
                ETHERTYPE_IPV6
            } else {
                ETHERTYPE_IPV4
            };
            (ethertype, frame)
        }
        LINKTYPE_NULL => {
            let version = frame.get(NULL_HEADER_BYTES)? >> 4;
            let ethertype = if version == 6 {
                ETHERTYPE_IPV6
            } else {
                ETHERTYPE_IPV4
            };
            (ethertype, frame.get(NULL_HEADER_BYTES..)?)
        }
        _ => return None,
    };
    let udp = match ethertype {
        ETHERTYPE_IPV4 => {
            let header = usize::from(ip.first()? & 0x0F) * 4;
            if *ip.get(9)? != IP_PROTOCOL_UDP || header < IPV4_HEADER_BYTES {
                return None;
            }
            let total = usize::from(be16(ip, 2)?).min(ip.len());
            ip.get(header..total)?
        }
        ETHERTYPE_IPV6 => {
            if *ip.get(6)? != IP_PROTOCOL_UDP {
                return None;
            }
            let payload = usize::from(be16(ip, 4)?);
            ip.get(IPV6_HEADER_BYTES..IPV6_HEADER_BYTES + payload)?
        }
        _ => return None,
    };
    let port = be16(udp, 2)?;
    let length = usize::from(be16(udp, 4)?).min(udp.len());
    Some((port, udp.get(UDP_HEADER_BYTES..length)?))
}

/// Append one packet in RFC 4571 framing.
///
/// # Errors
/// A packet over 65535 bytes, or the writer fails.
pub fn write_framed<W: Write>(out: &mut W, rtp: &[u8]) -> Result<()> {
    let length =
        u16::try_from(rtp.len()).map_err(|_| RtpError::Config("packet over 64 KiB".into()))?;
    out.write_all(&length.to_be_bytes())?;
    out.write_all(rtp)?;
    Ok(())
}

/// Read every packet of an RFC 4571 framed stream.
///
/// # Errors
/// A truncated packet, or the reader fails.
pub fn read_framed<R: Read>(mut input: R) -> Result<Vec<CapturedPacket>> {
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes)?;
    let mut packets = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let prefix = bytes
            .get(at..at + FRAMING_LENGTH_BYTES)
            .ok_or(RtpError::Truncated("RFC 4571 length"))?;
        let length = usize::from(u16::from_be_bytes([prefix[0], prefix[1]]));
        let start = at + FRAMING_LENGTH_BYTES;
        let data = bytes
            .get(start..start + length)
            .ok_or(RtpError::Truncated("RFC 4571 packet"))?;
        packets.push(CapturedPacket {
            time_us: 0,
            port: 0,
            data: data.to_vec(),
        });
        at = start + length;
    }
    Ok(packets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcap_round_trips() {
        let mut writer = PcapWriter::new(Vec::new()).unwrap();
        writer.write(1_500_000, &[0x80, 96, 0, 1]).unwrap();
        writer.write(1_533_333, &[0x80, 96, 0, 2, 9]).unwrap();
        let bytes = writer.into_inner();
        let packets = read_pcap(bytes.as_slice()).unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[1].data, vec![0x80, 96, 0, 2, 9]);
        assert_eq!((packets[0].time_us, packets[0].port), (1_500_000, RTP_PORT));
        // The IPv4 checksum verifies to zero.
        let ip = &bytes[24 + 16 + 14..24 + 16 + 14 + 20];
        assert_eq!(ipv4_checksum(ip), 0);
    }

    #[test]
    fn framed_streams_round_trip() {
        let mut out = Vec::new();
        write_framed(&mut out, &[1, 2, 3]).unwrap();
        write_framed(&mut out, &[4]).unwrap();
        let packets = read_framed(out.as_slice()).unwrap();
        assert_eq!(
            packets.iter().map(|p| p.data.clone()).collect::<Vec<_>>(),
            vec![vec![1, 2, 3], vec![4]]
        );
        assert!(read_framed(&[0u8, 5, 1][..]).is_err());
    }

    #[test]
    fn rtcp_is_told_apart() {
        assert!(is_rtcp(&[0x80, 200]));
        assert!(!is_rtcp(&[0x80, 96]));
        assert!(!is_rtcp(&[0x80, 0xE0])); // marker + PT 96
    }
}
