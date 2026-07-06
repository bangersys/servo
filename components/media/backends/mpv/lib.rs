mod events;
pub mod player;
mod render;
mod stream;

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, Weak};

use log::warn;
use servo_base::generic_channel::GenericCallback;
use servo_media::{Backend, BackendDeInit, BackendInit, MediaInstanceError, SupportsMediaType};
use servo_media_audio::context::{AudioContext, AudioContextOptions};
use servo_media_audio::decoder::{AudioDecoder, AudioDecoderCallbacks, AudioDecoderOptions};
use servo_media_audio::render_thread::AudioRenderThreadMsg;
use servo_media_audio::sink::{AudioSink, AudioSinkError};
use servo_media_audio::{AudioBackend, AudioStreamReader};
use servo_media_player::audio::AudioRenderer;
use servo_media_player::context::PlayerGLContext;
use servo_media_player::video::VideoFrameRenderer;
use servo_media_player::{Player, PlayerEvent, StreamType};
use servo_media_streams::capture::MediaTrackConstraintSet;
use servo_media_streams::device_monitor::{MediaDeviceInfo, MediaDeviceMonitor};
use servo_media_streams::registry::{MediaStreamId, register_stream, unregister_stream};
use servo_media_streams::{MediaOutput, MediaSocket, MediaStream, MediaStreamType};
use servo_media_traits::{BackendMsg, ClientContextId, MediaInstance};
use servo_media_webrtc::{
    BundlePolicy, DataChannelId, DataChannelInit, DataChannelMessage, IceCandidate,
    SessionDescription, WebRtcBackend, WebRtcController, WebRtcControllerBackend,
    WebRtcDataChannelResult, WebRtcResult, WebRtcSignaller, thread,
};

use crate::player::MpvPlayer;

pub struct MpvBackend {
    capture_mocking: AtomicBool,
    instances: Arc<Mutex<HashMap<ClientContextId, Vec<(usize, Weak<Mutex<dyn MediaInstance>>)>>>>,
    next_instance_id: AtomicUsize,
    backend_chan: Arc<Mutex<Sender<BackendMsg>>>,
}

impl MpvBackend {
    fn media_instance_action(
        &self,
        id: &ClientContextId,
        cb: &dyn Fn(&dyn MediaInstance) -> Result<(), MediaInstanceError>,
    ) {
        let mut instances = self.instances.lock().unwrap();
        match instances.get_mut(id) {
            Some(vec) => vec.retain(|(_, weak)| match weak.upgrade() {
                Some(instance) => {
                    if cb(&*(instance.lock().unwrap())).is_err() {
                        warn!("Error executing media instance action");
                    }
                    true
                },
                _ => false,
            }),
            None => {
                warn!("Trying to exec media action on an unknown client context");
            },
        }
    }
}

impl BackendInit for MpvBackend {
    fn init() -> Box<dyn Backend> {
        let instances: HashMap<ClientContextId, Vec<(usize, Weak<Mutex<dyn MediaInstance>>)>> =
            HashMap::new();
        let instances = Arc::new(Mutex::new(instances));

        let instances_ = instances.clone();
        let (backend_chan, recvr) = mpsc::channel();
        std::thread::Builder::new()
            .name("MpvBackend ShutdownThread".to_owned())
            .spawn(move || {
                loop {
                    match recvr.recv() {
                        Ok(BackendMsg::Shutdown {
                            context,
                            id,
                            tx_ack,
                        }) => {
                            let mut instances_ = instances_.lock().unwrap();
                            if let Some(vec) = instances_.get_mut(&context) {
                                vec.retain(|m| m.0 != id);
                                if vec.is_empty() {
                                    instances_.remove(&context);
                                }
                            }
                            let _ = tx_ack.send(());
                        },
                        Err(_) => break,
                    };
                }
            })
            .unwrap();

        Box::new(MpvBackend {
            capture_mocking: AtomicBool::new(false),
            instances,
            next_instance_id: AtomicUsize::new(0),
            backend_chan: Arc::new(Mutex::new(backend_chan)),
        })
    }
}

impl BackendDeInit for MpvBackend {
    fn deinit(&self) {
        let to_shutdown: Vec<(ClientContextId, usize)> = {
            let map = self.instances.lock().unwrap();
            map.iter()
                .flat_map(|(ctx, v)| v.iter().map(move |(id, _)| (*ctx, *id)))
                .collect()
        };

        for (ctx, id) in to_shutdown {
            let (tx_ack, rx_ack) = mpsc::channel();
            let _ = self
                .backend_chan
                .lock()
                .unwrap()
                .send(BackendMsg::Shutdown {
                    context: ctx,
                    id,
                    tx_ack,
                });
            let _ = rx_ack.recv();
        }
    }
}

impl Backend for MpvBackend {
    fn create_player(
        &self,
        context_id: &ClientContextId,
        stream_type: StreamType,
        sender: GenericCallback<PlayerEvent>,
        video_renderer: Option<Arc<Mutex<dyn VideoFrameRenderer>>>,
        audio_renderer: Option<Arc<Mutex<dyn AudioRenderer>>>,
        gl_context: Box<dyn PlayerGLContext>,
    ) -> Arc<Mutex<dyn Player>> {
        let id = self.next_instance_id.fetch_add(1, Ordering::Relaxed);
        let player = Arc::new(Mutex::new(MpvPlayer::new(
            id,
            context_id,
            self.backend_chan.clone(),
            stream_type,
            sender,
            video_renderer,
            audio_renderer,
            gl_context,
        )));
        let mut instances = self.instances.lock().unwrap();
        let entry = instances.entry(*context_id).or_default();
        let weak: Weak<Mutex<dyn MediaInstance>> =
            Arc::downgrade(&(player.clone() as Arc<Mutex<dyn MediaInstance>>));
        entry.push((id, weak));
        player
    }

    fn create_audiostream(&self) -> MediaStreamId {
        register_stream(Arc::new(Mutex::new(MpvMediaStream {
            id: MediaStreamId::new(),
        })))
    }

    fn create_videostream(&self) -> MediaStreamId {
        register_stream(Arc::new(Mutex::new(MpvMediaStream {
            id: MediaStreamId::new(),
        })))
    }

    fn create_stream_output(&self) -> Box<dyn MediaOutput> {
        Box::new(MpvMediaOutput)
    }

    fn create_audioinput_stream(&self, _set: MediaTrackConstraintSet) -> Option<MediaStreamId> {
        Some(register_stream(Arc::new(Mutex::new(MpvMediaStream {
            id: MediaStreamId::new(),
        }))))
    }

    fn create_videoinput_stream(&self, _set: MediaTrackConstraintSet) -> Option<MediaStreamId> {
        Some(register_stream(Arc::new(Mutex::new(MpvMediaStream {
            id: MediaStreamId::new(),
        }))))
    }

    fn create_stream_and_socket(
        &self,
        _ty: MediaStreamType,
    ) -> (Box<dyn MediaSocket>, MediaStreamId) {
        let id = register_stream(Arc::new(Mutex::new(MpvMediaStream {
            id: MediaStreamId::new(),
        })));
        (Box::new(MpvSocket), id)
    }

    fn create_audio_context(
        &self,
        client_context_id: &ClientContextId,
        options: AudioContextOptions,
    ) -> Result<Arc<Mutex<AudioContext>>, AudioSinkError> {
        let id = self.next_instance_id.fetch_add(1, Ordering::Relaxed);
        let audio_context =
            AudioContext::new::<Self>(id, client_context_id, self.backend_chan.clone(), options)?;
        let audio_context = Arc::new(Mutex::new(audio_context));
        let audio_context_dyn: Arc<Mutex<dyn MediaInstance>> = audio_context.clone();
        let mut instances = self.instances.lock().unwrap();
        let entry = instances.entry(*client_context_id).or_default();
        entry.push((id, Arc::downgrade(&audio_context_dyn)));
        Ok(audio_context)
    }

    fn create_webrtc(&self, signaller: Box<dyn WebRtcSignaller>) -> WebRtcController {
        WebRtcController::new::<Self>(signaller)
    }

    fn can_play_type(&self, _media_type: &str) -> SupportsMediaType {
        SupportsMediaType::Probably
    }

    fn set_capture_mocking(&self, mock: bool) {
        self.capture_mocking.store(mock, Ordering::Release)
    }

    fn mute(&self, id: &ClientContextId, val: bool) {
        self.media_instance_action(id, &move |instance: &dyn MediaInstance| instance.mute(val));
    }

    fn suspend(&self, id: &ClientContextId) {
        self.media_instance_action(id, &|instance: &dyn MediaInstance| instance.suspend());
    }

    fn resume(&self, id: &ClientContextId) {
        self.media_instance_action(id, &|instance: &dyn MediaInstance| instance.resume());
    }

    fn get_device_monitor(&self) -> Box<dyn MediaDeviceMonitor> {
        Box::new(MpvMediaDeviceMonitor)
    }
}

impl AudioBackend for MpvBackend {
    type Sink = MpvAudioSink;
    fn make_decoder() -> Box<dyn AudioDecoder> {
        Box::new(MpvAudioDecoder)
    }
    fn make_sink() -> Result<Self::Sink, AudioSinkError> {
        Ok(MpvAudioSink)
    }
    fn make_streamreader(
        _id: MediaStreamId,
        _sample_rate: f32,
    ) -> Result<Box<dyn AudioStreamReader + Send>, AudioSinkError> {
        Ok(Box::new(MpvStreamReader))
    }
}

impl WebRtcBackend for MpvBackend {
    type Controller = MpvWebRtcController;
    fn construct_webrtc_controller(
        _signaller: Box<dyn WebRtcSignaller>,
        _controller: WebRtcController,
    ) -> Self::Controller {
        MpvWebRtcController
    }
}

pub struct MpvAudioDecoder;
impl AudioDecoder for MpvAudioDecoder {
    fn decode(
        &self,
        _data: Vec<u8>,
        _callbacks: AudioDecoderCallbacks,
        _options: Option<AudioDecoderOptions>,
    ) {
    }
}

pub struct MpvAudioSink;
impl AudioSink for MpvAudioSink {
    fn init(
        &self,
        _sample_rate: f32,
        _sender: Sender<AudioRenderThreadMsg>,
    ) -> Result<(), AudioSinkError> {
        Ok(())
    }
    fn init_stream(
        &self,
        _channels: u8,
        _sample_rate: f32,
        _socket: Box<dyn MediaSocket>,
    ) -> Result<(), AudioSinkError> {
        Ok(())
    }
    fn play(&self) -> Result<(), AudioSinkError> {
        Ok(())
    }
    fn stop(&self) -> Result<(), AudioSinkError> {
        Ok(())
    }
    fn has_enough_data(&self) -> bool {
        true
    }
    fn push_data(&self, _chunk: servo_media_audio::block::Chunk) -> Result<(), AudioSinkError> {
        Ok(())
    }
    fn set_eos_callback(&self, _cb: Box<dyn Fn(Box<dyn AsRef<[f32]>>) + Send + Sync + 'static>) {}
}

pub struct MpvStreamReader;
impl AudioStreamReader for MpvStreamReader {
    fn pull(&self) -> servo_media_audio::block::Block {
        Default::default()
    }
    fn start(&self) {}
    fn stop(&self) {}
}

pub struct MpvMediaStream {
    id: MediaStreamId,
}

impl MediaStream for MpvMediaStream {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }
    fn set_id(&mut self, _id: MediaStreamId) {}
    fn ty(&self) -> MediaStreamType {
        MediaStreamType::Audio
    }
}

impl Drop for MpvMediaStream {
    fn drop(&mut self) {
        unregister_stream(&self.id);
    }
}

pub struct MpvSocket;
impl MediaSocket for MpvSocket {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct MpvMediaOutput;
impl MediaOutput for MpvMediaOutput {
    fn add_stream(&mut self, _stream: &MediaStreamId) {}
}

pub struct MpvWebRtcController;
impl WebRtcControllerBackend for MpvWebRtcController {
    fn configure(&mut self, _: &str, _: BundlePolicy) -> WebRtcResult {
        Ok(())
    }
    fn set_remote_description(
        &mut self,
        _: SessionDescription,
        _: Box<dyn FnOnce() + Send + 'static>,
    ) -> WebRtcResult {
        Ok(())
    }
    fn set_local_description(
        &mut self,
        _: SessionDescription,
        _: Box<dyn FnOnce() + Send + 'static>,
    ) -> WebRtcResult {
        Ok(())
    }
    fn add_ice_candidate(&mut self, _: IceCandidate) -> WebRtcResult {
        Ok(())
    }
    fn create_offer(
        &mut self,
        _: Box<dyn FnOnce(SessionDescription) + Send + 'static>,
    ) -> WebRtcResult {
        Ok(())
    }
    fn create_answer(
        &mut self,
        _: Box<dyn FnOnce(SessionDescription) + Send + 'static>,
    ) -> WebRtcResult {
        Ok(())
    }
    fn add_stream(&mut self, _: &MediaStreamId) -> WebRtcResult {
        Ok(())
    }
    fn create_data_channel(&mut self, _: &DataChannelInit) -> WebRtcDataChannelResult {
        Ok(0)
    }
    fn close_data_channel(&mut self, _: &DataChannelId) -> WebRtcResult {
        Ok(())
    }
    fn send_data_channel_message(
        &mut self,
        _: &DataChannelId,
        _: &DataChannelMessage,
    ) -> WebRtcResult {
        Ok(())
    }
    fn internal_event(&mut self, _: thread::InternalEvent) -> WebRtcResult {
        Ok(())
    }
    fn quit(&mut self) {}
}

struct MpvMediaDeviceMonitor;
impl MediaDeviceMonitor for MpvMediaDeviceMonitor {
    fn enumerate_devices(&self) -> Option<Vec<MediaDeviceInfo>> {
        Some(vec![])
    }
}
