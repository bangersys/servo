use std::collections::HashMap;
use std::os::raw as ctype;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use libmpv2::Mpv;
use libmpv2::protocol::Protocol;
use log::error;
use servo_base::generic_channel::{self, GenericCallback};
use servo_media_player::{PlayerEvent, SeekLock, SeekLockMsg};

const BUFFER_HIGH_WATER: usize = 10 * 1024 * 1024;
const BUFFER_LOW_WATER: usize = 1024 * 1024;
// How much already-downloaded data we keep behind the current read position.
// This lets small backward seeks (e.g. replaying the last few seconds) be
// served locally, while bounding memory use for long-running playback of
// large files. Seeks further back than this simply fall through to the
// network-seek path below.
const BUFFER_TRIM_MARGIN: u64 = 20 * 1024 * 1024;

struct StreamData {
    /// Bytes downloaded so far, starting at logical offset `buffer_start`.
    buffer: Vec<u8>,
    /// Logical file offset represented by `buffer[0]`.
    buffer_start: u64,
    /// Logical file offset of the next byte mpv will read.
    read_pos: u64,
    eof: bool,
    cancelled: bool,
    /// True while a network-level seek (SeekData round trip) is in flight.
    /// Data pushed while this is set belongs to the fetch we're abandoning
    /// and must be dropped, mirroring GStreamer's ServoSrc::push_buffer.
    seeking: bool,
    /// Known total size of the resource, if any (from set_input_size / a
    /// Content-Length header). Independent of how much we've downloaded.
    total_size: Option<u64>,
}

pub struct ServoStream {
    inner: Mutex<StreamData>,
    cvar: Condvar,
    /// `None` only for the defensive fallback stream constructed in
    /// `open_fn` if a stream ID can't be resolved (should not happen in
    /// normal operation -- see `register_protocol`). Without an observer we
    /// can't negotiate a network seek, so `seek()` falls back to clamping
    /// within whatever is locally buffered (which will be nothing).
    observer: Option<Arc<Mutex<GenericCallback<PlayerEvent>>>>,
}

impl ServoStream {
    pub fn new(observer: Option<Arc<Mutex<GenericCallback<PlayerEvent>>>>) -> Self {
        ServoStream {
            inner: Mutex::new(StreamData {
                buffer: Vec::with_capacity(1024 * 1024),
                buffer_start: 0,
                read_pos: 0,
                eof: false,
                cancelled: false,
                seeking: false,
                total_size: None,
            }),
            cvar: Condvar::new(),
            observer,
        }
    }

    pub fn push_data(&self, data: &[u8]) {
        let mut inner = self.inner.lock().unwrap();
        if inner.seeking {
            // A network seek is in progress; this data belongs to the fetch
            // we're abandoning. Drop it, same as GStreamer's ServoSrc does
            // while `seeking` is set.
            return;
        }
        inner.buffer.extend_from_slice(data);

        // Bound memory: drop history further behind read_pos than
        // BUFFER_TRIM_MARGIN. Any future seek into the trimmed range will
        // simply take the network-seek path instead of the local fast path.
        let trim_before = inner.read_pos.saturating_sub(BUFFER_TRIM_MARGIN);
        if trim_before > inner.buffer_start {
            let drop_count = (trim_before - inner.buffer_start) as usize;
            if drop_count > 0 && drop_count <= inner.buffer.len() {
                inner.buffer.drain(0..drop_count);
                inner.buffer_start += drop_count as u64;
            }
        }

        self.cvar.notify_all();
    }

    pub fn set_eof(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.eof = true;
        self.cvar.notify_all();
    }

    /// Records the resource's total size, e.g. from a Content-Length header.
    /// Lets mpv know the real duration/seek range before the whole file has
    /// downloaded, matching ServoSrc::set_size on the GStreamer side.
    pub fn set_size(&self, size: u64) {
        let mut inner = self.inner.lock().unwrap();
        inner.total_size = Some(size);
    }

    pub fn is_buffer_full(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        let write_pos = inner.buffer_start + inner.buffer.len() as u64;
        (write_pos.saturating_sub(inner.read_pos) as usize) >= BUFFER_HIGH_WATER
    }

    #[allow(dead_code)]
    pub fn is_buffer_low(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        let write_pos = inner.buffer_start + inner.buffer.len() as u64;
        (write_pos.saturating_sub(inner.read_pos) as usize) < BUFFER_LOW_WATER
    }

    pub fn cancel(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.cancelled = true;
        self.cvar.notify_all();
    }

    pub fn read(&self, buf: &mut [ctype::c_char]) -> i64 {
        let mut inner = self.inner.lock().unwrap();
        loop {
            let write_pos = inner.buffer_start + inner.buffer.len() as u64;
            if inner.read_pos < write_pos {
                break;
            }
            if inner.cancelled {
                return -1;
            }
            if inner.eof {
                return 0;
            }
            inner = self.cvar.wait(inner).unwrap();
        }

        let write_pos = inner.buffer_start + inner.buffer.len() as u64;
        let available = (write_pos - inner.read_pos) as usize;
        let to_read = available.min(buf.len());
        let start = (inner.read_pos - inner.buffer_start) as usize;

        let src = &inner.buffer[start..start + to_read];
        for (i, &byte) in src.iter().enumerate() {
            buf[i] = byte as ctype::c_char;
        }
        inner.read_pos += to_read as u64;
        to_read as i64
    }

    /// Called from mpv's demuxer thread whenever mpv needs to reposition the
    /// stream. Serves the seek locally if the target offset is already
    /// buffered; otherwise performs a real network seek by notifying the
    /// client via `PlayerEvent::SeekData` and blocking (mirroring
    /// GStreamer's ServoSrc::set_seek_offset / SeekData / SeekLock protocol)
    /// until the client has arranged a new ranged fetch at that offset.
    pub fn seek(&self, offset: i64) -> i64 {
        if offset < 0 {
            let inner = self.inner.lock().unwrap();
            return inner.read_pos as i64;
        }
        let offset = offset as u64;

        {
            let mut inner = self.inner.lock().unwrap();
            let write_pos = inner.buffer_start + inner.buffer.len() as u64;
            if offset >= inner.buffer_start && offset <= write_pos {
                // Already downloaded (or seeking exactly to the current
                // end) -- no network round trip needed.
                inner.read_pos = offset;
                return offset as i64;
            }
        }

        let Some(observer) = self.observer.as_ref() else {
            // No observer to negotiate a network seek with (defensive
            // fallback stream only) -- clamp locally as a last resort.
            let mut inner = self.inner.lock().unwrap();
            let write_pos = inner.buffer_start + inner.buffer.len() as u64;
            inner.read_pos = offset.min(write_pos).max(inner.buffer_start);
            return inner.read_pos as i64;
        };

        let (sender, ack_recv) = match generic_channel::channel::<SeekLockMsg>() {
            Some(pair) => pair,
            None => {
                error!("ServoStream: failed to create seek IPC channel");
                let inner = self.inner.lock().unwrap();
                return inner.read_pos as i64;
            },
        };
        let seek_lock = SeekLock {
            lock_channel: sender,
        };

        {
            let mut inner = self.inner.lock().unwrap();
            inner.seeking = true;
        }

        if observer
            .lock()
            .unwrap()
            .send(PlayerEvent::SeekData(offset, seek_lock))
            .is_err()
        {
            error!("ServoStream: failed to notify observer of SeekData");
            let mut inner = self.inner.lock().unwrap();
            inner.seeking = false;
            return inner.read_pos as i64;
        }

        // Block the demuxer thread until the client has arranged the new
        // ranged fetch and is ready for data to start flowing again.
        let ack_sender = match ack_recv.recv() {
            Ok((_result, ack_sender)) => ack_sender,
            Err(_) => {
                let mut inner = self.inner.lock().unwrap();
                inner.seeking = false;
                return inner.read_pos as i64;
            },
        };

        {
            let mut inner = self.inner.lock().unwrap();
            inner.buffer.clear();
            inner.buffer_start = offset;
            inner.read_pos = offset;
            inner.seeking = false;
        }
        self.cvar.notify_all();

        // Unblock the client, which was waiting for us to finish applying
        // the seek before it starts pushing data from the new offset.
        let _ = ack_sender.send(());

        offset as i64
    }

    pub fn size(&self) -> i64 {
        let inner = self.inner.lock().unwrap();
        if let Some(size) = inner.total_size {
            size as i64
        } else if inner.eof {
            (inner.buffer_start + inner.buffer.len() as u64) as i64
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

    pub fn register(
        &mut self,
        observer: Arc<Mutex<GenericCallback<PlayerEvent>>>,
    ) -> (u64, Arc<ServoStream>) {
        let id = self.next_id;
        self.next_id += 1;
        let stream = Arc::new(ServoStream::new(Some(observer)));
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

/// A single process-wide registry shared by all players. This is safe and
/// simple *because* protocol registration (see `register_protocol` below)
/// is now correctly scoped per mpv core: each player's mpv instance only
/// ever calls back into `open_fn` for stream IDs that it itself requested
/// via `loadfile "servo://<id>"`, so sharing one registry across players
/// introduces no cross-talk -- it just avoids needing a separate registry
/// object per player for no benefit.
static GLOBAL_REGISTRY: OnceLock<Arc<Mutex<StreamRegistry>>> = OnceLock::new();

pub fn global_registry() -> Arc<Mutex<StreamRegistry>> {
    GLOBAL_REGISTRY
        .get_or_init(|| Arc::new(Mutex::new(StreamRegistry::new())))
        .clone()
}

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
    // NOTE: this fallback empty stream is only reachable on a genuine
    // internal inconsistency (e.g. the ID was already removed), since
    // protocol registration is now scoped per player/core -- see
    // `register_protocol`. It has no observer, so seeks on it just clamp
    // locally instead of negotiating a network fetch.
    Arc::new(ServoStream::new(None))
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

/// Owns the per-player mpv protocol registration. Must be kept alive for as
/// long as the owning `MpvPlayer`'s mpv core is alive: dropping it frees the
/// callback table that mpv invokes into (see libmpv2's `Protocol`'s `Drop`).
pub struct ProtocolHandle {
    _proto: Protocol<'static, StreamCookie, StreamUserData>,
}

/// Registers the `servo://` stream protocol on `mpv`'s own core.
///
/// IMPORTANT: each `MpvPlayer` owns a fully separate mpv core (`Mpv::new()`
/// creates an independent `mpv_handle`/core, not a lightweight client of a
/// shared one). `Protocol::register()` wraps `mpv_stream_cb_add_ro`, which
/// registers the protocol against the specific core pointed to by
/// `self.mpv.ctx` -- it does NOT apply process-wide. This function must
/// therefore be called once per player, with that player's own `mpv`
/// handle.
///
/// (The previous implementation gated this behind a single process-wide
/// `OnceLock`, so only the very first player's core ever had `servo://`
/// registered at all; every subsequent player's `loadfile "servo://N"`
/// failed outright, which is the root cause of the repeated
/// `mpv event error: -13` seen after the first video.)
pub fn register_protocol(mpv: &Mpv, registry: StreamUserData) -> ProtocolHandle {
    // NOTE: create_client(None) is used because the safe wrapper's
    // create_client(Some(name)) has a temporary CString lifetime issue in
    // the Servo runtime environment (the FFI works fine with a long-lived
    // CString, but the wrapper's inline CString causes mpv_create_client to
    // return NULL on mpv 0.41.0).
    //
    // The client handle is intentionally leaked to obtain the `'static`
    // reference `Protocol` requires. This leaks one small mpv client handle
    // per player for the lifetime of the process; a follow-up could avoid
    // this by threading a non-'static lifetime through `MpvPlayer` instead.
    let client_mpv = mpv
        .create_client(None)
        .expect("Failed to create mpv client for protocol");
    let client_mpv: &'static Mpv = Box::leak(Box::new(client_mpv));

    let proto = unsafe {
        Protocol::new(
            client_mpv,
            "servo".to_owned(),
            registry,
            open_fn,
            close_fn,
            read_fn,
            Some(seek_fn),
            Some(size_fn),
        )
    };
    proto
        .register()
        .expect("Failed to register servo protocol");
    ProtocolHandle { _proto: proto }
}