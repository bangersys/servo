use std::collections::HashMap;
use std::os::raw as ctype;
use std::sync::{Arc, Condvar, Mutex};

use libmpv2::Mpv;
use libmpv2::protocol::Protocol;
use log::error;

const BUFFER_HIGH_WATER: usize = 10 * 1024 * 1024;
const BUFFER_LOW_WATER: usize = 1024 * 1024;

struct StreamData {
    buffer: Vec<u8>,
    read_pos: usize,
    eof: bool,
    cancelled: bool,
}

pub struct ServoStream {
    inner: Mutex<StreamData>,
    cvar: Condvar,
}

impl ServoStream {
    pub fn new() -> Self {
        ServoStream {
            inner: Mutex::new(StreamData {
                buffer: Vec::with_capacity(1024 * 1024),
                read_pos: 0,
                eof: false,
                cancelled: false,
            }),
            cvar: Condvar::new(),
        }
    }

    pub fn push_data(&self, data: &[u8]) {
        let mut inner = self.inner.lock().unwrap();
        inner.buffer.extend_from_slice(data);
        self.cvar.notify_all();
    }

    pub fn set_eof(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.eof = true;
        self.cvar.notify_all();
    }

    pub fn is_buffer_full(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.buffer.len() - inner.read_pos >= BUFFER_HIGH_WATER
    }

    #[allow(dead_code)]
    pub fn is_buffer_low(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.buffer.len() - inner.read_pos < BUFFER_LOW_WATER
    }

    pub fn cancel(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.cancelled = true;
        self.cvar.notify_all();
    }

    pub fn read(&self, buf: &mut [ctype::c_char]) -> i64 {
        let mut inner = self.inner.lock().unwrap();
        while inner.read_pos >= inner.buffer.len() {
            if inner.cancelled {
                return -1;
            }
            if inner.eof {
                return 0;
            }
            inner = self.cvar.wait(inner).unwrap();
        }

        let available = inner.buffer.len() - inner.read_pos;
        let to_read = available.min(buf.len());

        let src = &inner.buffer[inner.read_pos..inner.read_pos + to_read];
        for (i, &byte) in src.iter().enumerate() {
            buf[i] = byte as ctype::c_char;
        }
        inner.read_pos += to_read;
        to_read as i64
    }

    pub fn seek(&self, offset: i64) -> i64 {
        let mut inner = self.inner.lock().unwrap();
        if offset < 0 {
            return inner.read_pos as i64;
        }
        inner.read_pos = offset as usize;
        if inner.read_pos > inner.buffer.len() {
            inner.read_pos = inner.buffer.len();
        }
        inner.read_pos as i64
    }

    pub fn size(&self) -> i64 {
        let inner = self.inner.lock().unwrap();
        if inner.eof {
            inner.buffer.len() as i64
        } else {
            -1
        }
    }
}

pub struct StreamRegistry {
    streams: HashMap<u64, Arc<ServoStream>>,
    next_id: u64,
}

impl StreamRegistry {
    pub fn new() -> Self {
        StreamRegistry {
            streams: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn register(&mut self) -> (u64, Arc<ServoStream>) {
        let id = self.next_id;
        self.next_id += 1;
        let stream = Arc::new(ServoStream::new());
        self.streams.insert(id, stream.clone());
        (id, stream)
    }

    pub fn get(&mut self, id: u64) -> Option<Arc<ServoStream>> {
        self.streams.get(&id).cloned()
    }

    pub fn remove(&mut self, id: u64) {
        self.streams.remove(&id);
    }
}

type StreamCookie = Arc<ServoStream>;
type StreamUserData = Arc<Mutex<StreamRegistry>>;

fn open_fn(user_data: &mut StreamUserData, uri: &str) -> StreamCookie {
    let id_str = uri.trim_start_matches("servo://");
    match id_str.parse::<u64>() {
        Ok(id) => {
            if let Some(stream) = user_data.lock().unwrap().get(id) {
                return stream;
            }
            error!("ServoStream: no stream found for ID {id}");
        },
        Err(_) => {
            error!("ServoStream: cannot parse stream ID from URI: {uri}");
        },
    }
    Arc::new(ServoStream::new())
}

fn read_fn(stream: &mut StreamCookie, buf: &mut [ctype::c_char]) -> i64 {
    stream.read(buf)
}

fn seek_fn(stream: &mut StreamCookie, offset: i64) -> i64 {
    stream.seek(offset)
}

fn size_fn(stream: &mut StreamCookie) -> i64 {
    stream.size()
}

fn close_fn(stream: Box<StreamCookie>) {
    drop(stream);
}

use std::sync::OnceLock;

static PROTO: OnceLock<Protocol<'static, StreamCookie, StreamUserData>> = OnceLock::new();

pub fn ensure_protocol(mpv: &Mpv, registry: StreamUserData) {
    PROTO.get_or_init(|| {
        // NOTE: create_client(None) is used because the safe wrapper's
        // create_client(Some(name)) has a temporary CString lifetime issue
        // in the Servo runtime environment (the FFI works fine with a
        // long-lived CString, but the wrapper's inline CString causes
        // mpv_create_client to return NULL on mpv 0.41.0).
        let client_mpv = mpv
            .create_client(None)
            .expect("Failed to create mpv client for protocol");
        let _client_mpv: &'static Mpv = Box::leak(Box::new(client_mpv));

        let proto = unsafe {
            Protocol::new(
                _client_mpv,
                "servo".to_owned(),
                registry,
                open_fn,
                close_fn,
                read_fn,
                Some(seek_fn),
                Some(size_fn),
            )
        };
        proto.register().expect("Failed to register servo protocol");
        proto
    });
}
