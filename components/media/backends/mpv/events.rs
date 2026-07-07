use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use libmpv2::events::{Event, PropertyData};
use libmpv2::{Error, Mpv, mpv_end_file_reason};
use log::warn;
use servo_base::generic_channel::GenericCallback;
use servo_media_player::metadata::Metadata;
use servo_media_player::{PlaybackState, PlayerEvent};

pub fn start_event_loop(mpv: Arc<Mpv>, observer: Arc<Mutex<GenericCallback<PlayerEvent>>>) {
    let _ = thread::Builder::new()
        .name("MpvEventLoop".to_owned())
        .spawn(move || {
            loop {
                match mpv.wait_event(0.1) {
                    Some(Ok(event)) => handle_event(&mpv, &observer, event),
                    Some(Err(Error::Raw(code))) if code != 0 => {
                        log::error!("mpv event error: {code}");
                        thread::sleep(Duration::from_millis(50));
                    },
                    Some(Err(_)) => {},
                    None => {},
                }
            }
        })
        .expect("Thread spawning failed");
}

fn handle_event(
    mpv: &Mpv,
    observer: &Arc<std::sync::Mutex<GenericCallback<PlayerEvent>>>,
    event: Event,
) {
    match event {
        Event::EndFile(reason) => {
            warn!("mpv event: EndFile reason={}", reason as i32);
            if reason == mpv_end_file_reason::Eof {
                notify(observer, PlayerEvent::EndOfStream);
            } else if reason == mpv_end_file_reason::Error {
                notify(observer, PlayerEvent::Error("Playback error".to_owned()));
            }
        },
        Event::StartFile => {
            warn!("mpv event: StartFile");
        },
        Event::FileLoaded => {
            warn!("mpv event: FileLoaded");
            let duration = mpv.get_property::<f64>("duration").ok().and_then(|d| {
                if d > 0.0 {
                    let secs = d.trunc() as u64;
                    let nanos = ((d - d.trunc()) * 1_000_000_000.0) as u32;
                    Some(Duration::new(secs, nanos))
                } else {
                    None
                }
            });
            let width = mpv.get_property::<i64>("width").unwrap_or(0) as u32;
            let height = mpv.get_property::<i64>("height").unwrap_or(0) as u32;
            let metadata = Metadata {
                duration,
                width,
                height,
                format: String::new(),
                is_seekable: mpv.get_property::<bool>("seekable").unwrap_or(false),
                audio_tracks: vec![],
                video_tracks: vec![],
                is_live: false,
                title: None,
            };
            notify(observer, PlayerEvent::MetadataUpdated(metadata));
            notify(observer, PlayerEvent::DurationChanged(duration));
            // Must be sent after MetadataUpdated so that the HTMLMediaElement's
            // ready_state has reached HaveMetadata when this event is processed.
            // This triggers the autoplay path: StateChanged → playback_state_changed
            // → change_ready_state(HaveEnoughData) → eligible_for_autoplay → play().
            notify(observer, PlayerEvent::StateChanged(PlaybackState::Paused));
            notify(observer, PlayerEvent::VideoFrameUpdated);
        },
        Event::PropertyChange { name, change, .. } => match name {
            "time-pos" => {
                if let PropertyData::Double(pos) = change {
                    notify(observer, PlayerEvent::PositionChanged(pos));
                }
            },
            "pause" => {
                if let PropertyData::Flag(paused) = change {
                    let state = if paused {
                        PlaybackState::Paused
                    } else {
                        PlaybackState::Playing
                    };
                    notify(observer, PlayerEvent::StateChanged(state));
                }
            },
            _ => {},
        },
        Event::Seek => {
            if let Ok(pos) = mpv.get_property::<f64>("time-pos") {
                notify(observer, PlayerEvent::SeekDone(pos));
            }
        },
        Event::VideoReconfig => {
            warn!("mpv event: VideoReconfig");
            let width = mpv.get_property::<i64>("width").unwrap_or(0) as u32;
            let height = mpv.get_property::<i64>("height").unwrap_or(0) as u32;
            notify(
                observer,
                PlayerEvent::MetadataUpdated(Metadata {
                    duration: None,
                    width,
                    height,
                    format: String::new(),
                    is_seekable: mpv.get_property::<bool>("seekable").unwrap_or(false),
                    audio_tracks: vec![],
                    video_tracks: vec![],
                    is_live: false,
                    title: None,
                }),
            );
            notify(observer, PlayerEvent::VideoFrameUpdated);
        },
        _ => {},
    }
}

fn notify(observer: &Arc<std::sync::Mutex<GenericCallback<PlayerEvent>>>, event: PlayerEvent) {
    if let Ok(guard) = observer.lock() {
        let _ = guard.send(event);
    }
}
