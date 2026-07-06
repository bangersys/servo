use std::ops::Range;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};

use libmpv2::{Format, Mpv};
use servo_base::generic_channel::GenericCallback;
use servo_media::MediaInstanceError;
use servo_media_player::audio::AudioRenderer;
use servo_media_player::context::PlayerGLContext;
use servo_media_player::video::VideoFrameRenderer;
use servo_media_player::{Player, PlayerError, PlayerEvent, StreamType};
use servo_media_streams::registry::MediaStreamId;
use servo_media_traits::{BackendMsg, ClientContextId, MediaInstance};

use crate::events::start_event_loop;
use crate::render::{self, RenderCommand, RenderHandle};
use crate::stream::{self, ServoStream, StreamRegistry};

const DEFAULT_MUTED: bool = false;
const DEFAULT_PAUSED: bool = true;
const DEFAULT_VOLUME: f64 = 1.0;

pub struct MpvPlayer {
    id: usize,
    context_id: ClientContextId,
    backend_chan: Arc<Mutex<Sender<BackendMsg>>>,
    mpv: Arc<Mpv>,
    render_handle: Option<RenderHandle>,
    stream: Arc<ServoStream>,
    stream_id: u64,
    stream_type: StreamType,
    stream_registry: Arc<Mutex<StreamRegistry>>,
    loaded: Mutex<bool>,
}

impl MpvPlayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: usize,
        context_id: &ClientContextId,
        backend_chan: Arc<Mutex<Sender<BackendMsg>>>,
        stream_type: StreamType,
        observer: GenericCallback<PlayerEvent>,
        video_renderer: Option<Arc<Mutex<dyn VideoFrameRenderer>>>,
        _audio_renderer: Option<Arc<Mutex<dyn AudioRenderer>>>,
        gl_context: Box<dyn PlayerGLContext>,
    ) -> Self {
        let mpv = Mpv::new().expect("Failed to create mpv instance");

        mpv.enable_all_events().ok();
        mpv.set_property("vo", "libmpv").ok();
        mpv.set_property("keep-open", false).ok();
        mpv.set_property("mute", DEFAULT_MUTED).ok();
        mpv.set_property("volume", DEFAULT_VOLUME * 100.0).ok();
        mpv.set_property("pause", DEFAULT_PAUSED).ok();

        mpv.observe_property("time-pos", Format::Double, 0).ok();
        mpv.observe_property("pause", Format::Flag, 0).ok();

        let registry = Arc::new(Mutex::new(StreamRegistry::new()));
        let (stream_id, stream) = registry.lock().unwrap().register();
        stream::ensure_protocol(&mpv, registry.clone());

        let mpv_arc = Arc::new(mpv);
        let observer = Arc::new(Mutex::new(observer));
        start_event_loop(mpv_arc.clone(), observer.clone());

        let render_handle =
            render::spawn_render_thread(mpv_arc.clone(), gl_context, video_renderer, observer);

        MpvPlayer {
            id,
            context_id: *context_id,
            backend_chan,
            mpv: mpv_arc,
            render_handle: Some(render_handle),
            stream,
            stream_id,
            stream_type,
            stream_registry: registry,
            loaded: Mutex::new(false),
        }
    }

    fn load_if_needed(&self) -> Result<(), PlayerError> {
        let mut loaded = self.loaded.lock().unwrap();
        if !*loaded {
            self.mpv
                .command("loadfile", &[&format!("servo://{}", self.stream_id)])
                .map_err(|e| PlayerError::Backend(format!("mpv loadfile failed: {e}")))?;
            *loaded = true;
        }
        Ok(())
    }
}

impl Player for MpvPlayer {
    fn play(&self) -> Result<(), PlayerError> {
        self.load_if_needed()?;
        self.mpv
            .set_property("pause", false)
            .map_err(|e| PlayerError::Backend(format!("mpv play failed: {e}")))
    }

    fn pause(&self) -> Result<(), PlayerError> {
        self.mpv
            .set_property("pause", true)
            .map_err(|e| PlayerError::Backend(format!("mpv pause failed: {e}")))
    }

    fn paused(&self) -> bool {
        self.mpv.get_property::<bool>("pause").unwrap_or(true)
    }

    fn can_resume(&self) -> bool {
        !self.paused()
    }

    fn stop(&self) -> Result<(), PlayerError> {
        self.mpv
            .command("stop", &[])
            .map_err(|e| PlayerError::Backend(format!("mpv stop failed: {e}")))
    }

    fn seek(&self, time: f64) -> Result<(), PlayerError> {
        self.load_if_needed()?;
        self.mpv
            .set_property("time-pos", time)
            .map_err(|e| PlayerError::Backend(format!("mpv seek failed: {e}")))
    }

    fn seekable(&self) -> Vec<Range<f64>> {
        if self.stream_type == StreamType::Stream {
            return vec![];
        }
        if let Ok(dur) = self.mpv.get_property::<f64>("duration") {
            if dur > 0.0 {
                return vec![Range {
                    start: 0.0,
                    end: dur,
                }];
            }
        }
        vec![]
    }

    fn set_mute(&self, muted: bool) -> Result<(), PlayerError> {
        self.mpv
            .set_property("mute", muted)
            .map_err(|e| PlayerError::Backend(format!("mpv set_mute failed: {e}")))
    }

    fn muted(&self) -> bool {
        self.mpv.get_property::<bool>("mute").unwrap_or(false)
    }

    fn set_volume(&self, volume: f64) -> Result<(), PlayerError> {
        self.mpv
            .set_property("volume", volume * 100.0)
            .map_err(|e| PlayerError::Backend(format!("mpv set_volume failed: {e}")))
    }

    fn volume(&self) -> f64 {
        self.mpv.get_property::<f64>("volume").unwrap_or(100.0) / 100.0
    }

    fn set_input_size(&self, _size: u64) -> Result<(), PlayerError> {
        Ok(())
    }

    fn set_seekable(&self, _seekable: bool) -> Result<(), PlayerError> {
        Ok(())
    }

    fn set_playback_rate(&self, rate: f64) -> Result<(), PlayerError> {
        self.mpv
            .set_property("speed", rate)
            .map_err(|e| PlayerError::Backend(format!("mpv set_playback_rate failed: {e}")))
    }

    fn playback_rate(&self) -> f64 {
        self.mpv.get_property::<f64>("speed").unwrap_or(1.0)
    }

    fn push_data(&self, data: Vec<u8>) -> Result<(), PlayerError> {
        self.load_if_needed()?;
        self.stream.push_data(&data);
        if self.stream.is_buffer_full() {
            Err(PlayerError::EnoughData)
        } else {
            Ok(())
        }
    }

    fn end_of_stream(&self) -> Result<(), PlayerError> {
        self.stream.set_eof();
        Ok(())
    }

    fn buffered(&self) -> Vec<Range<f64>> {
        vec![]
    }

    fn set_stream(&self, _stream: &MediaStreamId, _only_stream: bool) -> Result<(), PlayerError> {
        Err(PlayerError::SetStreamFailed)
    }

    fn render_use_gl(&self) -> bool {
        self.render_handle.as_ref().is_some_and(|h| h.is_gl)
    }

    fn set_audio_track(&self, _stream_index: i32, _enabled: bool) -> Result<(), PlayerError> {
        Ok(())
    }

    fn set_video_track(&self, _stream_index: i32, _enabled: bool) -> Result<(), PlayerError> {
        Ok(())
    }
}

impl MediaInstance for MpvPlayer {
    fn get_id(&self) -> usize {
        self.id
    }

    fn mute(&self, val: bool) -> Result<(), MediaInstanceError> {
        self.set_mute(val).map_err(|_| MediaInstanceError)
    }

    fn suspend(&self) -> Result<(), MediaInstanceError> {
        self.pause().map_err(|_| MediaInstanceError)
    }

    fn resume(&self) -> Result<(), MediaInstanceError> {
        if !self.can_resume() {
            return Ok(());
        }
        self.play().map_err(|_| MediaInstanceError)
    }
}

impl Drop for MpvPlayer {
    fn drop(&mut self) {
        // 1. Signal render thread shutdown and join it FIRST
        //    (RenderContext must be freed while GL context is still valid)
        if let Some(handle) = self.render_handle.take() {
            let _ = handle.shutdown_tx.send(RenderCommand::Shutdown);
            if let Some(thread) = handle.thread {
                let _ = thread.join();
            }
        }
        // 2. Existing cleanup
        self.stream.cancel();
        let _ = self.mpv.command("stop", &[]);
        self.stream_registry.lock().unwrap().remove(self.stream_id);
        // 3. Existing Shutdown/ack
        let (tx_ack, rx_ack) = mpsc::channel();
        let _ = self
            .backend_chan
            .lock()
            .unwrap()
            .send(BackendMsg::Shutdown {
                context: self.context_id,
                id: self.id,
                tx_ack,
            });
        let _ = rx_ack.recv();
    }
}
