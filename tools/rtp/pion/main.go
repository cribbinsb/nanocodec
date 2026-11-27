// Command rtpinterop drives Pion's RTP payloaders and depacketizers
// (github.com/pion/rtp, MIT licence) for the nanocodec RTP interop tests
// (docs/handoff/rtp.md, tools/rtp/interop.py): an independent
// implementation of the same payload formats, both directions.
//
//	rtpinterop pay   --codec C --in STREAM --out OUT.rtp [--mtu N]
//	rtpinterop depay --codec C --in IN.rtp --out STREAM
//	rtpinterop webrtc ...   (webrtc.go: the Chrome end-to-end check)
//
// Codecs: h264, hevc, vvc (Annex B), av1, vp9 (IVF). RTP files use RFC 4571
// framing (a 16-bit length before each packet).
package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/pion/rtp"
	"github.com/pion/rtp/codecs"
)

const (
	clockRate   = 90000
	payloadType = 96
	ssrc        = 0x5049_4f4e
	ivfHeader   = 32
	ivfFrameHdr = 12
	frameTicks  = 3000 // 30 fps in 90 kHz ticks
)

func main() {
	if len(os.Args) < 2 {
		fail(errors.New("usage: rtpinterop pay|depay --codec C --in IN --out OUT"))
	}
	command := os.Args[1]
	if command == "webrtc" {
		if err := runWebRTC(os.Args[2:]); err != nil {
			fail(err)
		}
		return
	}
	flags := flag.NewFlagSet(command, flag.ExitOnError)
	codec := flags.String("codec", "", "h264, hevc, vvc, av1 or vp9")
	in := flags.String("in", "", "input path")
	out := flags.String("out", "", "output path")
	mtu := flags.Int("mtu", 1200, "largest RTP packet")
	flexible := flags.Bool("vp9-flexible", false, "VP9 flexible mode")
	_ = flags.Parse(os.Args[2:])
	var err error
	switch command {
	case "pay":
		err = pay(*codec, *in, *out, uint16(*mtu), *flexible)
	case "depay":
		err = depay(*codec, *in, *out)
	default:
		err = fmt.Errorf("unknown command %q", command)
	}
	if err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "rtpinterop:", err)
	os.Exit(1)
}

// units reads a stream file as decoder units: IVF frames, or Annex-B access
// units (a new unit before the first NAL after a VCL NAL: one slice per
// picture, as nanocodec writes).
func units(codec string, data []byte) ([][]byte, error) {
	switch codec {
	case "av1", "vp9":
		if len(data) < ivfHeader || string(data[:4]) != "DKIF" {
			return nil, errors.New("not an IVF file")
		}
		var frames [][]byte
		for at := ivfHeader; at < len(data); {
			if at+ivfFrameHdr > len(data) {
				return nil, errors.New("truncated IVF frame header")
			}
			size := int(binary.LittleEndian.Uint32(data[at:]))
			at += ivfFrameHdr
			if at+size > len(data) {
				return nil, errors.New("truncated IVF frame")
			}
			frames = append(frames, data[at:at+size])
			at += size
		}
		return frames, nil
	default:
		return accessUnits(codec, data), nil
	}
}

func isVCL(codec string, nal []byte) bool {
	switch codec {
	case "h264":
		t := nal[0] & 0x1f
		return t >= 1 && t <= 5
	case "hevc":
		return (nal[0]>>1)&0x3f <= 31
	default: // vvc
		return len(nal) > 1 && nal[1]>>3 <= 11
	}
}

func accessUnits(codec string, data []byte) [][]byte {
	var starts []int
	for i := 0; i+3 <= len(data); i++ {
		if data[i] == 0 && data[i+1] == 0 && data[i+2] == 1 {
			start := i
			if i > 0 && data[i-1] == 0 {
				start = i - 1
			}
			starts = append(starts, start)
			i += 2
		}
	}
	var units [][]byte
	unitStart := 0
	previousVCL := false
	for index, start := range starts {
		end := len(data)
		if index+1 < len(starts) {
			end = starts[index+1]
		}
		nal := bytes.TrimLeft(data[start:end], "\x00")[1:]
		if previousVCL && index > 0 {
			units = append(units, data[unitStart:start])
			unitStart = start
		}
		previousVCL = isVCL(codec, nal)
	}
	if len(starts) > 0 {
		units = append(units, data[unitStart:])
	}
	return units
}

func payloader(codec string, flexible bool) (rtp.Payloader, error) {
	switch codec {
	case "h264":
		return &codecs.H264Payloader{}, nil
	case "hevc":
		return &codecs.H265Payloader{}, nil
	case "vvc":
		return &codecs.H266Packetizer{}, nil
	case "av1":
		return &codecs.AV1Payloader{}, nil
	case "vp9":
		return &codecs.VP9Payloader{FlexibleMode: flexible}, nil
	}
	return nil, fmt.Errorf("unknown codec %q", codec)
}

func pay(codec, in, out string, mtu uint16, flexible bool) error {
	data, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	list, err := units(codec, data)
	if err != nil {
		return err
	}
	p, err := payloader(codec, flexible)
	if err != nil {
		return err
	}
	packetizer := rtp.NewPacketizer(mtu, payloadType, ssrc, p, rtp.NewFixedSequencer(1000), clockRate)
	var buffer bytes.Buffer
	count := 0
	for index, unit := range list {
		samples := uint32(frameTicks)
		if index == 0 {
			samples = 0
		}
		for _, packet := range packetizer.Packetize(unit, samples) {
			raw, err := packet.Marshal()
			if err != nil {
				return err
			}
			_ = binary.Write(&buffer, binary.BigEndian, uint16(len(raw)))
			buffer.Write(raw)
			count++
		}
	}
	fmt.Fprintf(os.Stderr, "pion pay: %d units, %d packets\n", len(list), count)
	return os.WriteFile(out, buffer.Bytes(), 0o644)
}

type depacketizer interface {
	Unmarshal([]byte) ([]byte, error)
}

func newDepacketizer(codec string) (depacketizer, error) {
	switch codec {
	case "h264":
		return &codecs.H264Packet{}, nil
	case "hevc":
		return &codecs.H265Depacketizer{}, nil
	case "vvc":
		return &codecs.H266Depacketizer{}, nil
	case "av1":
		return &codecs.AV1Depacketizer{}, nil
	case "vp9":
		return &codecs.VP9Packet{}, nil
	}
	return nil, fmt.Errorf("unknown codec %q", codec)
}

// temporalDelimiter is an AV1 temporal delimiter OBU with its size field.
var temporalDelimiter = []byte{0x12, 0x00}

func depay(codec, in, out string) error {
	data, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	d, err := newDepacketizer(codec)
	if err != nil {
		return err
	}
	var output bytes.Buffer
	ivf := codec == "av1" || codec == "vp9"
	if ivf {
		writeIVFHeader(&output, codec)
	}
	var unit []byte
	// VP9: where each layer frame of the picture starts (B bit), so a
	// picture of several frames is written as a superframe (VP9 Annex B).
	var frameStarts []int
	var unitTimestamp uint32
	var firstTimestamp *uint32
	units := 0
	flush := func() {
		if len(unit) == 0 {
			return
		}
		if codec == "av1" && !bytes.HasPrefix(unit, temporalDelimiter) {
			unit = append(append([]byte{}, temporalDelimiter...), unit...)
		}
		if codec == "vp9" && len(frameStarts) > 1 {
			unit = appendSuperframeIndex(unit, frameStarts)
		}
		frameStarts = nil
		if ivf {
			if firstTimestamp == nil {
				t := unitTimestamp
				firstTimestamp = &t
			}
			_ = binary.Write(&output, binary.LittleEndian, uint32(len(unit)))
			_ = binary.Write(&output, binary.LittleEndian, uint64(unitTimestamp-*firstTimestamp))
		}
		output.Write(unit)
		unit = nil
		units++
	}
	reader := bytes.NewReader(data)
	for {
		var length uint16
		if err := binary.Read(reader, binary.BigEndian, &length); err == io.EOF {
			break
		} else if err != nil {
			return err
		}
		raw := make([]byte, length)
		if _, err := io.ReadFull(reader, raw); err != nil {
			return err
		}
		var packet rtp.Packet
		if err := packet.Unmarshal(raw); err != nil {
			return err
		}
		if len(unit) > 0 && packet.Timestamp != unitTimestamp {
			flush()
		}
		unitTimestamp = packet.Timestamp
		if codec == "vp9" {
			// Pion's VP9Packet keeps P_DIFFs across Unmarshal calls (it
			// appends without resetting), so a reused one fails on the
			// fourth reference: parse each packet with a fresh one.
			d = &codecs.VP9Packet{}
		}
		piece, err := d.Unmarshal(packet.Payload)
		if err != nil {
			return fmt.Errorf("packet %d: %w", packet.SequenceNumber, err)
		}
		if vp9, ok := d.(*codecs.VP9Packet); ok && vp9.B {
			frameStarts = append(frameStarts, len(unit))
		}
		unit = append(unit, piece...)
		if packet.Marker {
			flush()
		}
	}
	flush()
	fmt.Fprintf(os.Stderr, "pion depay: %d units\n", units)
	return os.WriteFile(out, output.Bytes(), 0o644)
}

// appendSuperframeIndex appends the VP9 superframe index (Annex B) of the
// frames starting at starts: marker, 4-byte little-endian sizes, marker.
func appendSuperframeIndex(unit []byte, starts []int) []byte {
	marker := byte(0xc0 | 3<<3 | (len(starts) - 1))
	unit = append(unit, marker)
	frames := len(unit) - 1
	for i, start := range starts {
		end := frames
		if i+1 < len(starts) {
			end = starts[i+1]
		}
		unit = binary.LittleEndian.AppendUint32(unit, uint32(end-start))
	}
	return append(unit, marker)
}

func writeIVFHeader(w *bytes.Buffer, codec string) {
	fourcc := "AV01"
	if codec == "vp9" {
		fourcc = "VP90"
	}
	w.WriteString("DKIF")
	_ = binary.Write(w, binary.LittleEndian, uint16(0))
	_ = binary.Write(w, binary.LittleEndian, uint16(ivfHeader))
	w.WriteString(fourcc)
	_ = binary.Write(w, binary.LittleEndian, uint16(0)) // width: frames carry their own
	_ = binary.Write(w, binary.LittleEndian, uint16(0))
	_ = binary.Write(w, binary.LittleEndian, uint32(clockRate))
	_ = binary.Write(w, binary.LittleEndian, uint32(1))
	_ = binary.Write(w, binary.LittleEndian, uint32(0))
	_ = binary.Write(w, binary.LittleEndian, uint32(0))
}
