/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Mpv GL and software render paths. Creates a dedicated per-player render
//! thread with either a shared EGL context + FBO-based texture output (GL) or
//! a CPU-side pixel buffer (software).

use std::ffi::c_void;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use khronos_egl as egl;
use libmpv2::Mpv;
use libmpv2::render::mpv_render_update;
use libmpv2::render::{OpenGLInitParams, RenderParam, RenderParamApiType};
use log::{error, warn};
use servo_base::generic_channel::GenericCallback;
use servo_media_player::PlayerEvent;
use servo_media_player::context::{GlApi, GlContext, NativeDisplay, PlayerGLContext};
use servo_media_player::video::{Buffer, VideoFrame, VideoFrameData, VideoFrameRenderer};

// ---------------------------------------------------------------------------
// GL type aliases
// ---------------------------------------------------------------------------

type GLuint = u32;
type GLint = i32;
type GLenum = u32;
type GLsizei = i32;

const GL_FRAMEBUFFER: GLenum = 0x8D40;
const GL_TEXTURE_2D: GLenum = 0x0DE1;
const GL_COLOR_ATTACHMENT0: GLenum = 0x8CE0;
const GL_RGBA: GLenum = 0x1908;
const GL_UNSIGNED_BYTE: GLenum = 0x1401;

// ---------------------------------------------------------------------------
// GL function pointer table – loaded via eglGetProcAddress
// ---------------------------------------------------------------------------

struct GlFunctions {
    gl_gen_framebuffers: unsafe extern "system" fn(GLsizei, *mut GLuint),
    gl_delete_framebuffers: unsafe extern "system" fn(GLsizei, *const GLuint),
    gl_bind_framebuffer: unsafe extern "system" fn(GLenum, GLuint),
    gl_framebuffer_texture_2d: unsafe extern "system" fn(GLenum, GLenum, GLenum, GLuint, GLint),
    gl_gen_textures: unsafe extern "system" fn(GLsizei, *mut GLuint),
    gl_delete_textures: unsafe extern "system" fn(GLsizei, *const GLuint),
    gl_bind_texture: unsafe extern "system" fn(GLenum, GLuint),
    gl_tex_image_2d: unsafe extern "system" fn(
        GLenum,
        GLint,
        GLint,
        GLsizei,
        GLsizei,
        GLint,
        GLenum,
        GLenum,
        *const c_void,
    ),
    gl_viewport: unsafe extern "system" fn(GLint, GLint, GLsizei, GLsizei),
}

#[allow(clippy::missing_transmute_annotations)]
fn load_gl_functions(egl: &egl::DynamicInstance<egl::EGL1_4>) -> Option<GlFunctions> {
    macro_rules! load {
        ($name:expr) => {
            match egl.get_proc_address($name) {
                Some(f) => unsafe { std::mem::transmute::<extern "system" fn(), _>(f) },
                None => {
                    error!("render_thread: failed to load GL function: {}", $name);
                    return None;
                },
            }
        };
    }

    Some(GlFunctions {
        gl_gen_framebuffers: load!("glGenFramebuffers"),
        gl_delete_framebuffers: load!("glDeleteFramebuffers"),
        gl_bind_framebuffer: load!("glBindFramebuffer"),
        gl_framebuffer_texture_2d: load!("glFramebufferTexture2D"),
        gl_gen_textures: load!("glGenTextures"),
        gl_delete_textures: load!("glDeleteTextures"),
        gl_bind_texture: load!("glBindTexture"),
        gl_tex_image_2d: load!("glTexImage2D"),
        gl_viewport: load!("glViewport"),
    })
}

// ---------------------------------------------------------------------------
// MpvBuffer – passed to VideoFrame for texture output
// ---------------------------------------------------------------------------

pub struct MpvBuffer {
    pub texture_id: u32,
    #[allow(dead_code)]
    pub width: i32,
    #[allow(dead_code)]
    pub height: i32,
}

impl Buffer for MpvBuffer {
    fn to_vec(&self) -> Option<VideoFrameData> {
        Some(VideoFrameData::Texture(self.texture_id))
    }
}

// ---------------------------------------------------------------------------
// MpvGlCtx – unit struct used as the GLContext generic parameter for mpv's
// OpenGL render API. Holds a raw pointer to the EGL instance.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct MpvGlCtx {
    egl: *const c_void,
}

unsafe impl Send for MpvGlCtx {}
unsafe impl Sync for MpvGlCtx {}

fn mpv_get_proc_address(ctx: &MpvGlCtx, name: &str) -> *mut c_void {
    let inst = unsafe { &*(ctx.egl as *const egl::DynamicInstance<egl::EGL1_4>) };
    inst.get_proc_address(name)
        .map(|f| f as *mut c_void)
        .unwrap_or(std::ptr::null_mut())
}

// ---------------------------------------------------------------------------
// Software buffer – holds CPU-side pixel data (BGRA8)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub struct SwBuffer {
    pub data: Arc<Vec<u8>>,
    pub width: i32,
    pub height: i32,
}

impl Buffer for SwBuffer {
    fn to_vec(&self) -> Option<VideoFrameData> {
        Some(VideoFrameData::Raw(self.data.clone()))
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub enum RenderCommand {
    Wakeup,
    Shutdown,
}

pub struct RenderHandle {
    pub shutdown_tx: Sender<RenderCommand>,
    pub thread: Option<JoinHandle<()>>,
    pub is_gl: bool,
}

// ---------------------------------------------------------------------------
// Entry point – spawns the dedicated render thread
// ---------------------------------------------------------------------------

pub fn spawn_render_thread(
    mpv: Arc<Mpv>,
    gl_context: Box<dyn PlayerGLContext>,
    video_renderer: Option<Arc<Mutex<dyn VideoFrameRenderer>>>,
    observer: Arc<Mutex<GenericCallback<PlayerEvent>>>,
) -> RenderHandle {
    let (tx, rx) = mpsc::channel::<RenderCommand>();

    let native_display = gl_context.get_native_display();
    let gl_ctx = gl_context.get_gl_context();
    let gl_api = gl_context.get_gl_api();

    match (native_display, gl_ctx) {
        (NativeDisplay::Egl(d), GlContext::Egl(c)) => {
            // ---------- EGL / OpenGL path ----------
            spawn_gl_render_thread(mpv, d, c, gl_api, video_renderer, observer, tx, rx)
        },
        _ => {
            warn!("spawn_render_thread: non-EGL context, falling back to software render");
            spawn_sw_render_thread(mpv, video_renderer, observer, tx, rx)
        },
    }
}

/// OpenGL / EGL render thread – shared context, FBO output.
#[allow(clippy::too_many_arguments)]
fn spawn_gl_render_thread(
    mpv: Arc<Mpv>,
    egl_display_ptr: usize,
    app_egl_context_ptr: usize,
    gl_api: GlApi,
    video_renderer: Option<Arc<Mutex<dyn VideoFrameRenderer>>>,
    observer: Arc<Mutex<GenericCallback<PlayerEvent>>>,
    tx: Sender<RenderCommand>,
    rx: Receiver<RenderCommand>,
) -> RenderHandle {
    let wakeup_tx = tx.clone();

    let thread = match thread::Builder::new()
        .name("MpvRenderGL".into())
        .spawn(move || {
            gl_render_thread_main(
                mpv,
                egl_display_ptr,
                app_egl_context_ptr,
                gl_api,
                video_renderer,
                observer,
                rx,
                wakeup_tx,
            );
        }) {
        Ok(h) => Some(h),
        Err(e) => {
            error!("spawn_gl_render_thread: failed to spawn thread: {e}");
            return RenderHandle {
                shutdown_tx: tx,
                thread: None,
                is_gl: false,
            };
        },
    };

    RenderHandle {
        shutdown_tx: tx,
        thread,
        is_gl: true,
    }
}

/// Software render thread – CPU-side pixel buffer.
fn spawn_sw_render_thread(
    mpv: Arc<Mpv>,
    video_renderer: Option<Arc<Mutex<dyn VideoFrameRenderer>>>,
    observer: Arc<Mutex<GenericCallback<PlayerEvent>>>,
    tx: Sender<RenderCommand>,
    rx: Receiver<RenderCommand>,
) -> RenderHandle {
    let wakeup_tx = tx.clone();

    let thread = match thread::Builder::new()
        .name("MpvRenderSw".into())
        .spawn(move || {
            sw_render_thread_main(mpv, video_renderer, observer, rx, wakeup_tx);
        }) {
        Ok(h) => Some(h),
        Err(e) => {
            error!("spawn_sw_render_thread: failed to spawn thread: {e}");
            return RenderHandle {
                shutdown_tx: tx,
                thread: None,
                is_gl: false,
            };
        },
    };

    RenderHandle {
        shutdown_tx: tx,
        thread,
        is_gl: false,
    }
}

// ---------------------------------------------------------------------------
// GL render thread body
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn gl_render_thread_main(
    mpv: Arc<Mpv>,
    egl_display_ptr: usize,
    app_egl_context_ptr: usize,
    _gl_api: GlApi,
    video_renderer: Option<Arc<Mutex<dyn VideoFrameRenderer>>>,
    observer: Arc<Mutex<GenericCallback<PlayerEvent>>>,
    rx: Receiver<RenderCommand>,
    wakeup_tx: Sender<RenderCommand>,
) {
    // ---------- 1. Load EGL dynamically ----------
    let lib = match unsafe { libloading::Library::new("libEGL.so.1") } {
        Ok(l) => l,
        Err(e) => {
            warn!("render_thread: unable to load libEGL.so.1: {e}");
            return;
        },
    };

    let egl_inst = match unsafe { egl::DynamicInstance::<egl::EGL1_4>::load_required_from(lib) } {
        Ok(e) => e,
        Err(e) => {
            warn!("render_thread: unable to load EGL API: {e}");
            return;
        },
    };

    // ---------- 2. Get display ----------
    let display = match unsafe { egl_inst.get_display(egl_display_ptr as egl::NativeDisplayType) } {
        Some(d) => d,
        None => {
            warn!("render_thread: eglGetDisplay failed");
            return;
        },
    };

    // ---------- 3. Initialize display ----------
    if let Err(e) = egl_inst.initialize(display) {
        warn!("render_thread: eglInitialize: {e:?}");
        return;
    }

    // ---------- 4. Choose config ----------
    let config_attribs: &[egl::Int] = &[
        egl::SURFACE_TYPE as egl::Int,
        egl::PBUFFER_BIT as egl::Int,
        egl::RENDERABLE_TYPE as egl::Int,
        egl::OPENGL_ES_BIT as egl::Int,
        egl::RED_SIZE as egl::Int,
        8,
        egl::GREEN_SIZE as egl::Int,
        8,
        egl::BLUE_SIZE as egl::Int,
        8,
        egl::ALPHA_SIZE as egl::Int,
        8,
        egl::NONE as egl::Int,
    ];

    let config = match egl_inst.choose_first_config(display, config_attribs) {
        Ok(Some(c)) => c,
        Ok(None) => {
            warn!("render_thread: eglChooseFirstConfig returned no config");
            return;
        },
        Err(e) => {
            warn!("render_thread: eglChooseFirstConfig: {e:?}");
            return;
        },
    };

    // ---------- 5. Bind API ----------
    if egl_inst.bind_api(egl::OPENGL_ES_API).is_err() && egl_inst.bind_api(egl::OPENGL_API).is_err()
    {
        warn!("render_thread: eglBindAPI failed");
        return;
    }

    // ---------- 6. Create shared EGL context ----------
    let app_egl_context = unsafe { egl::Context::from_ptr(app_egl_context_ptr as egl::EGLContext) };

    let ctx_attribs: &[egl::Int] = &[
        egl::CONTEXT_MAJOR_VERSION as egl::Int,
        2,
        egl::CONTEXT_MINOR_VERSION as egl::Int,
        0,
        egl::NONE as egl::Int,
    ];

    let context = match egl_inst.create_context(display, config, Some(app_egl_context), ctx_attribs)
    {
        Ok(c) => c,
        Err(e) => {
            warn!("render_thread: eglCreateContext: {e:?}");
            return;
        },
    };

    // ---------- 7. Create pbuffer surface ----------
    let surf_attribs: &[egl::Int] = &[
        egl::WIDTH as egl::Int,
        1,
        egl::HEIGHT as egl::Int,
        1,
        egl::NONE as egl::Int,
    ];

    let surface = match egl_inst.create_pbuffer_surface(display, config, surf_attribs) {
        Ok(s) => s,
        Err(e) => {
            warn!("render_thread: eglCreatePbufferSurface: {e:?}");
            return;
        },
    };

    // ---------- 8. Make context current ----------
    if let Err(e) = egl_inst.make_current(display, Some(surface), Some(surface), Some(context)) {
        warn!("render_thread: eglMakeCurrent: {e:?}");
        return;
    }

    // ---------- 9. Load GL functions ----------
    let gl = match load_gl_functions(&egl_inst) {
        Some(g) => g,
        None => return,
    };

    // ---------- 10. Create MpvGlCtx ----------
    let mpv_gl_ctx = MpvGlCtx {
        egl: &egl_inst as *const _ as *const c_void,
    };

    // ---------- 11. Create mpv RenderContext ----------
    let mut render_ctx = match mpv.create_render_context(vec![
        RenderParam::ApiType(RenderParamApiType::OpenGl),
        RenderParam::InitParams(OpenGLInitParams {
            get_proc_address: mpv_get_proc_address,
            ctx: mpv_gl_ctx,
        }),
    ]) {
        Ok(ctx) => ctx,
        Err(e) => {
            warn!("render_thread: mpv create_render_context: {e}");
            return;
        },
    };

    // ---------- 12. Set update callback ----------
    render_ctx.set_update_callback(move || {
        let _ = wakeup_tx.send(RenderCommand::Wakeup);
    });

    // ---------- 13. Render loop ----------
    let mut fbo_pool: Vec<(GLuint, GLuint)> = Vec::new();
    let mut current_width: i32 = 0;
    let mut current_height: i32 = 0;

    'render: loop {
        let cmd = match rx.recv() {
            Ok(c) => c,
            Err(_) => break 'render,
        };

        match cmd {
            RenderCommand::Shutdown => break 'render,
            RenderCommand::Wakeup => {},
        }

        let flags = match render_ctx.update() {
            Ok(f) => f,
            Err(e) => {
                // update() can fail transiently if there's nothing to update
                error!("render_thread: mpv render_context_update: {e}");
                continue;
            },
        };

        if flags & mpv_render_update::Frame == 0 {
            continue;
        }

        let width = mpv.get_property::<i64>("width").unwrap_or(0) as i32;
        let height = mpv.get_property::<i64>("height").unwrap_or(0) as i32;

        if width <= 0 || height <= 0 {
            continue;
        }

        // Ensure FBO pool matches the current video size
        if width != current_width || height != current_height || fbo_pool.is_empty() {
            for &(tex, fbo) in &fbo_pool {
                unsafe {
                    (gl.gl_delete_framebuffers)(1, &fbo as *const GLuint);
                    (gl.gl_delete_textures)(1, &tex as *const GLuint);
                }
            }
            fbo_pool.clear();

            let mut fbo: GLuint = 0;
            let mut tex: GLuint = 0;

            unsafe {
                (gl.gl_gen_framebuffers)(1, &mut fbo as *mut GLuint);
                (gl.gl_gen_textures)(1, &mut tex as *mut GLuint);

                (gl.gl_bind_texture)(GL_TEXTURE_2D, tex);
                (gl.gl_tex_image_2d)(
                    GL_TEXTURE_2D,
                    0,
                    GL_RGBA as GLint,
                    width,
                    height,
                    0,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    std::ptr::null(),
                );

                (gl.gl_bind_framebuffer)(GL_FRAMEBUFFER, fbo);
                (gl.gl_framebuffer_texture_2d)(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_2D,
                    tex,
                    0,
                );

                (gl.gl_bind_framebuffer)(GL_FRAMEBUFFER, 0);
                (gl.gl_bind_texture)(GL_TEXTURE_2D, 0);
            }

            fbo_pool.push((tex, fbo));
            current_width = width;
            current_height = height;
        }

        let (tex_id, fbo_id) = fbo_pool[0];

        unsafe {
            (gl.gl_viewport)(0, 0, width, height);
        }

        if let Err(e) = render_ctx.render::<MpvGlCtx>(fbo_id as i32, width, height, true) {
            error!("render_thread: mpv render: {e}");
            continue;
        }

        let frame = VideoFrame::new(
            width,
            height,
            Arc::new(MpvBuffer {
                texture_id: tex_id,
                width,
                height,
            }),
        );

        if let Some(ref video_renderer) = video_renderer
            && let Some(frame) = frame
            && let Ok(mut guard) = video_renderer.lock()
        {
            guard.render(frame);
        }

        if let Ok(guard) = observer.lock() {
            let _ = guard.send(PlayerEvent::VideoFrameUpdated);
        }
    }

    // ---------- 14. Cleanup ----------
    drop(render_ctx);

    for &(tex, fbo) in &fbo_pool {
        unsafe {
            (gl.gl_delete_framebuffers)(1, &fbo as *const GLuint);
            (gl.gl_delete_textures)(1, &tex as *const GLuint);
        }
    }

    let _ = egl_inst.make_current(display, None, None, None);
    let _ = egl_inst.destroy_surface(display, surface);
    let _ = egl_inst.destroy_context(display, context);
    let _ = egl_inst.terminate(display);
}

// ---------------------------------------------------------------------------
// Software render thread body
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn sw_render_thread_main(
    mpv: Arc<Mpv>,
    video_renderer: Option<Arc<Mutex<dyn VideoFrameRenderer>>>,
    observer: Arc<Mutex<GenericCallback<PlayerEvent>>>,
    rx: Receiver<RenderCommand>,
    wakeup_tx: Sender<RenderCommand>,
) {
    // ---------- 1. Create mpv RenderContext with API type "sw" ----------
    let api_type = libmpv2_sys::MPV_RENDER_API_TYPE_SW.as_ptr() as *mut std::ffi::c_void;

    let raw_params = vec![
        libmpv2_sys::mpv_render_param {
            type_: libmpv2_sys::mpv_render_param_type_MPV_RENDER_PARAM_API_TYPE,
            data: api_type,
        },
        libmpv2_sys::mpv_render_param {
            type_: 0,
            data: std::ptr::null_mut(),
        },
    ];

    let raw_array =
        Box::into_raw(raw_params.into_boxed_slice()) as *mut libmpv2_sys::mpv_render_param;

    let mut ctx_ptr: *mut libmpv2_sys::mpv_render_context = std::ptr::null_mut();
    let result = unsafe {
        libmpv2_sys::mpv_render_context_create(
            &mut ctx_ptr as *mut *mut libmpv2_sys::mpv_render_context,
            mpv.ctx.as_ptr(),
            raw_array,
        )
    };
    unsafe {
        drop(Box::from_raw(raw_array));
    }

    if result != 0 {
        warn!("sw_render_thread: mpv_render_context_create failed: {result}");
        return;
    }
    warn!("sw_render_thread: mpv_render_context_create OK");

    let render_ctx: Box<libmpv2_sys::mpv_render_context> = unsafe { Box::from_raw(ctx_ptr) };
    // We need a raw pointer we can pass through FFI for the update callback.
    let ctx_raw: *mut libmpv2_sys::mpv_render_context = Box::into_raw(render_ctx);

    // ---------- 2. Set update callback ----------
    unsafe extern "C" fn update_callback(ctx: *mut std::ffi::c_void) {
        let sender = ctx as *const Sender<RenderCommand>;
        let _ = unsafe { (*sender).send(RenderCommand::Wakeup) };
    }

    let sender_for_mpv: *mut Sender<RenderCommand> = Box::into_raw(Box::new(wakeup_tx.clone()));

    unsafe {
        libmpv2_sys::mpv_render_context_set_update_callback(
            ctx_raw,
            Some(update_callback),
            sender_for_mpv as *mut std::ffi::c_void,
        );
    }

    // ---------- 3. Render loop ----------
    warn!("sw_render_thread: entering render loop");
    let mut pixel_data: Vec<u8> = Vec::new();
    let mut current_width: i32 = 0;
    let mut current_height: i32 = 0;

    'render: loop {
        let cmd = match rx.recv() {
            Ok(c) => c,
            Err(_) => {
                warn!("sw_render_thread: channel closed, exiting");
                break 'render;
            },
        };

        match cmd {
            RenderCommand::Shutdown => {
                warn!("sw_render_thread: Shutdown received");
                break 'render;
            },
            RenderCommand::Wakeup => {},
        }

        let flags = unsafe { libmpv2_sys::mpv_render_context_update(ctx_raw) };

        if flags & (libmpv2_sys::mpv_render_update_flag_MPV_RENDER_UPDATE_FRAME as u64) == 0 {
            warn!("sw_render_thread: update returned flags={flags}, no frame");
            continue;
        }

        let width = mpv.get_property::<i64>("width").unwrap_or(0) as i32;
        let height = mpv.get_property::<i64>("height").unwrap_or(0) as i32;

        if width <= 0 || height <= 0 {
            warn!("sw_render_thread: invalid dimensions {width}x{height}");
            continue;
        }

        warn!("sw_render_thread: rendering frame {width}x{height}");

        // Allocate or grow the pixel buffer when the video size changes.
        if width != current_width || height != current_height {
            let stride = width * 4; // BGRA8 = 4 bytes per pixel
            let size = (stride * height) as usize;
            pixel_data.resize(size, 0u8);
            current_width = width;
            current_height = height;
        }

        let mut stride = width * 4; // BGRA8 = 4 bytes per pixel

        let mut render_params: Vec<libmpv2_sys::mpv_render_param> = Vec::with_capacity(5);
        // SW_SIZE: int[2] = {width, height}
        let mut sw_size: [i32; 2] = [width, height];
        render_params.push(libmpv2_sys::mpv_render_param {
            type_: libmpv2_sys::mpv_render_param_type_MPV_RENDER_PARAM_SW_SIZE,
            data: &mut sw_size as *mut [i32; 2] as *mut std::ffi::c_void,
        });
        // SW_FORMAT: "bgra"
        let format = b"bgra\0";
        render_params.push(libmpv2_sys::mpv_render_param {
            type_: libmpv2_sys::mpv_render_param_type_MPV_RENDER_PARAM_SW_FORMAT,
            data: format.as_ptr() as *mut std::ffi::c_void,
        });
        // SW_STRIDE: int
        render_params.push(libmpv2_sys::mpv_render_param {
            type_: libmpv2_sys::mpv_render_param_type_MPV_RENDER_PARAM_SW_STRIDE,
            data: &mut stride as *mut i32 as *mut std::ffi::c_void,
        });
        // SW_POINTER: void*
        render_params.push(libmpv2_sys::mpv_render_param {
            type_: libmpv2_sys::mpv_render_param_type_MPV_RENDER_PARAM_SW_POINTER,
            data: pixel_data.as_mut_ptr() as *mut std::ffi::c_void,
        });
        // terminator
        render_params.push(libmpv2_sys::mpv_render_param {
            type_: 0,
            data: std::ptr::null_mut(),
        });

        let render_array =
            Box::into_raw(render_params.into_boxed_slice()) as *mut libmpv2_sys::mpv_render_param;

        let ret = unsafe { libmpv2_sys::mpv_render_context_render(ctx_raw, render_array) };
        unsafe {
            drop(Box::from_raw(render_array));
        }

        if ret != 0 {
            error!("sw_render_thread: mpv_render_context_render failed: {ret}");
            continue;
        }

        let frame = VideoFrame::new(
            width,
            height,
            Arc::new(SwBuffer {
                data: Arc::new(pixel_data.clone()),
                width,
                height,
            }),
        );

        if let Some(ref video_renderer) = video_renderer
            && let Some(frame) = frame
            && let Ok(mut guard) = video_renderer.lock()
        {
            guard.render(frame);
            warn!("sw_render_thread: frame rendered and sent to VideoFrameRenderer");
        } else {
            warn!("sw_render_thread: frame NOT rendered (missing VideoFrameRenderer)");
        }

        if let Ok(guard) = observer.lock() {
            let _ = guard.send(PlayerEvent::VideoFrameUpdated);
            warn!("sw_render_thread: VideoFrameUpdated sent");
        }
    }

    warn!("sw_render_thread: exiting");
    // ---------- 4. Cleanup ----------
    unsafe {
        libmpv2_sys::mpv_render_context_free(ctx_raw);
        drop(Box::from_raw(sender_for_mpv));
    }
}
