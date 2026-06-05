use std::io::{self, Read, Write};

pub(crate) fn write_u32(w: &mut dyn Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

pub(crate) fn write_u64(w: &mut dyn Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

pub(crate) fn write_i64(w: &mut dyn Write, v: i64) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

pub(crate) fn write_f32(w: &mut dyn Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_be_bytes())
}

pub(crate) fn write_u8(w: &mut dyn Write, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

pub(crate) fn read_u32(r: &mut dyn Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

pub(crate) fn read_u64(r: &mut dyn Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

pub(crate) fn read_i64(r: &mut dyn Read) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_be_bytes(buf))
}

pub(crate) fn read_f32(r: &mut dyn Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_be_bytes(buf))
}

pub(crate) fn read_u8(r: &mut dyn Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub(crate) fn write_f32_slice(w: &mut dyn Write, data: &[f32]) -> io::Result<()> {
    for &v in data {
        w.write_all(&v.to_be_bytes())?;
    }
    Ok(())
}

pub(crate) fn read_f32_vec(r: &mut dyn Read, len: usize) -> io::Result<Vec<f32>> {
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(read_f32(r)?);
    }
    Ok(vec)
}

pub(crate) fn write_byte_vec(w: &mut dyn Write, data: &[u8]) -> io::Result<()> {
    write_u32(w, data.len() as u32)?;
    w.write_all(data)
}

pub(crate) fn read_byte_vec(r: &mut dyn Read) -> io::Result<Vec<u8>> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}
