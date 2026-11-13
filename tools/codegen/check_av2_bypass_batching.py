#!/usr/bin/env python3
"""Isolated AV2 bypass batching experiment; production sources are never edited.

Compile original pinned AVM and a temporary Rust crate containing the current
entropy source with configurable Writer::literal chunk size. Check bytes against AVM and
native scalar output, malformed inputs, and existing entropy/model tests.
Reported normalize-call counts are structural, not measured throughput.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PIN = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
C = r'''
#include <stdio.h>
#include <stdint.h>
#include "avm_dsp/bitwriter.h"
int main(void) {
  uint8_t output[262144];
  avm_writer writer={0}; avm_start_encode(&writer,output);
  writer.allow_update_cdf=1;
  avm_cdf_prob cdf[]={30000,16000,5000,0,0,2,3,4};
  uint32_t state=12345;
  for(unsigned i=0;i<8192;i++) {
    state=state*1664525U+1013904223U;
    avm_write_symbol(&writer,state>>30,cdf,4);
    unsigned bits=i%33;
    uint32_t value=bits==32?state:state&((1U<<bits)-1);
    while(bits) {
      unsigned n=bits<8?bits:8;
      od_ec_encode_literal_bypass(&writer.ec,(value>>(bits-n))&((1U<<n)-1),n);
      bits-=n;
    }
  }
  avm_stop_encode(&writer);
  fwrite(output,1,writer.pos,stdout);
}
'''
BATCHED = '''        let mut remaining = bits;
        while remaining != 0 {
            let count = remaining.min(8);
            let chunk = (value >> (remaining - count)) & ((1 << count) - 1);
            self.low = (self.low << count) + u64::from(self.range) * u64::from(chunk);
            self.normalize(count as i32);
            remaining -= count;
        }
'''
SCALAR = '''        for index in (0..bits).rev() {
            self.low = (self.low << 1) + u64::from(self.range) * u64::from((value >> index) & 1);
            self.normalize(1);
        }
'''
RUST = r'''
#![allow(dead_code)]
#[path="ENTROPY"] mod entropy;
#[path="MODELS"] mod cdf_models;
#[cfg(test)] mod experiment {
use super::entropy::{Cdf, Reader, Writer};
use nc_av2::entropy as scalar;
fn scalar_literal(writer:&mut scalar::Writer,value:u32,bits:u32) {
    for index in (0..bits).rev() { writer.literal((value>>index)&1,1).unwrap(); }
}
fn values() -> impl Iterator<Item=(usize,u32,u32)> {
    let mut state=12345_u32;
    (0..8192).map(move |index| {
        state=state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let bits=index%33;
        let value=if bits==32 {state} else {state & ((1<<bits)-1)};
        ((state>>30) as usize,value,bits)
    })
}
#[test] fn original_c_bytes_and_scalar_bytes_match_all_widths() {
    let mut candidate=Writer::new(); let mut reference=scalar::Writer::new();
    let mut cdf=Cdf::new([30000,16000,5000,0],[2,3,4]).unwrap();
    let mut reference_cdf=scalar::Cdf::new([30000,16000,5000,0],[2,3,4]).unwrap();
    let mut scalar_calls=0; let mut batched_calls=0;
    for (symbol,value,bits) in values() {
        candidate.symbol(symbol,&mut cdf).unwrap(); candidate.literal(value,bits).unwrap();
        reference.symbol(symbol,&mut reference_cdf).unwrap(); scalar_literal(&mut reference,value,bits);
        scalar_calls+=bits; batched_calls+=bits.div_ceil(8);
    }
    let packet=candidate.finish(); assert_eq!(packet,reference.finish());
    assert_eq!(packet,include_bytes!("ORACLE").as_slice());
    let mut reader=Reader::new(&packet).unwrap();
    let mut cdf=Cdf::new([30000,16000,5000,0],[2,3,4]).unwrap();
    for (symbol,value,bits) in values() {
        assert_eq!(reader.symbol(&mut cdf).unwrap(),symbol);
        assert_eq!(reader.literal(bits).unwrap(),value);
    }
    eprintln!("Exact {} bytes; literal normalize calls {} scalar / {} batched (not timing)",packet.len(),scalar_calls,batched_calls);
}
#[test] fn widths_values_and_invalid_calls_preserve_writer_state() {
    for bits in 0..=32 {
        let maximum=if bits==32 {u32::MAX} else {(1<<bits)-1};
        for value in [0, maximum, maximum/2] {
            for prefix in 0..16 {
                let mut candidate=Writer::new(); let mut reference=scalar::Writer::new();
                candidate.literal(0,prefix).unwrap(); scalar_literal(&mut reference,0,prefix);
                assert!(candidate.literal(0,33).is_err());
                if bits<32 { assert!(candidate.literal(maximum+1,bits).is_err()); }
                candidate.literal(value,bits).unwrap(); scalar_literal(&mut reference,value,bits);
                assert_eq!(candidate.finish(),reference.finish());
            }
        }
    }
}
#[test] fn bounded_truncation_and_mutation_have_identical_decoder_outcomes() {
    let original=include_bytes!("ORACLE");
    for length in 0..=256 {
        let packet=&original[..length];
        let left=std::panic::catch_unwind(|| decode_candidate(packet));
        let right=std::panic::catch_unwind(|| decode_scalar(packet));
        assert_eq!(left.unwrap(),right.unwrap());
    }
    for bit in 0..512 {
        let mut packet=original[..128].to_vec(); packet[bit/8]^=1<<(bit%8);
        assert_eq!(decode_candidate(&packet),decode_scalar(&packet));
    }
    for bytes in [&[][..], &[0][..], &[255][..]] {
        if let Ok(mut reader)=Reader::new(bytes) { assert!(reader.literal(33).is_err()); }
    }
}
fn decode_candidate(packet:&[u8])->Result<Vec<u32>,String> {
    let mut r=Reader::new(packet).map_err(|e|format!("{e:?}"))?;
    let mut c=Cdf::new([30000,16000,5000,0],[2,3,4]).unwrap(); let mut result=Vec::new();
    for (_,_,bits) in values().take(256) {
        result.push(r.symbol(&mut c).map_err(|e|format!("{e:?}"))? as u32);
        result.push(r.literal(bits).map_err(|e|format!("{e:?}"))?);
    } Ok(result)
}
fn decode_scalar(packet:&[u8])->Result<Vec<u32>,String> {
    let mut r=scalar::Reader::new(packet).map_err(|e|format!("{e:?}"))?;
    let mut c=scalar::Cdf::new([30000,16000,5000,0],[2,3,4]).unwrap(); let mut result=Vec::new();
    for (_,_,bits) in values().take(256) {
        result.push(r.symbol(&mut c).map_err(|e|format!("{e:?}"))? as u32);
        result.push(r.literal(bits).map_err(|e|format!("{e:?}"))?);
    } Ok(result)
}
}
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--chunk',type=int,choices=[8,16],default=8)
    parser.add_argument('--target-dir',type=Path,default=Path.home()/'nanocodec-data/targets/vp9-av2-motion')
    args=parser.parse_args()
    assert subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()==PIN
    source=(ROOT/'crates/nc-av2/src/entropy.rs').read_text()
    production_loop = SCALAR if SCALAR in source else BATCHED.replace('remaining.min(8)', 'remaining.min(16)')
    assert source.count(production_loop)==1, 'Production literal changed; review this experiment'
    candidate=source.replace(production_loop,BATCHED.replace('remaining.min(8)',f'remaining.min({args.chunk})')).replace('include_bytes!("../tests/data/avm-entropy.bin")',f'include_bytes!("{ROOT}/crates/nc-av2/tests/data/avm-entropy.bin")')
    with tempfile.TemporaryDirectory(prefix='nc-av2-bypass-') as directory:
        path=Path(directory); (path/'oracle.c').write_text(C)
        subprocess.run(['cc','-O1','-I'+str(args.source),'-I'+str(args.build),str(path/'oracle.c'),str(args.build/'libavm.a'),'-lm','-lpthread','-o',str(path/'oracle')],check=True)
        (path/'oracle.bin').write_bytes(subprocess.check_output([str(path/'oracle')]))
        (path/'entropy.rs').write_text(candidate)
        (path/'lib.rs').write_text(RUST.replace('bits.div_ceil(8)',f'bits.div_ceil({args.chunk})').replace('ENTROPY',str(path/'entropy.rs')).replace('MODELS',str(ROOT/'crates/nc-av2/src/cdf_models.rs')).replace('ORACLE',str(path/'oracle.bin')))
        (path/'Cargo.toml').write_text(f'''[package]
name="nc-av2-bypass-experiment"
version="0.0.0"
edition="2024"
[lib]
path="lib.rs"
[dependencies]
nc-av2={{path="{ROOT}/crates/nc-av2"}}
nc-common={{path="{ROOT}/crates/nc-common"}}
''')
        env=os.environ.copy();env['CARGO_TARGET_DIR']=str(args.target_dir)
        subprocess.run([str(Path.home()/'.cargo/bin/cargo'),'test','--offline','--manifest-path',str(path/'Cargo.toml'),'-j2','--','--nocapture'],env=env,check=True)

if __name__=='__main__':main()
