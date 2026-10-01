#ifdef ENABLE_SOFTRAST

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#ifndef _LANGUAGE_C
# define _LANGUAGE_C
#endif
#include <PR/gbi.h>

#include "gfx_pc.h"
#include "gfx_soft.h"
#include "gfx_cc.h"
#include "macros.h"

#include "../savestate.h"

#define ALIGN(x, a) (((x) + (a - 1)) & ~(a - 1))

#define MAX_TEXTURES 3072
#define TEXCACHE_STEP 0x10000

enum WrapType {
    WRAP_REPEAT = 0,
    WRAP_CLAMP  = 1,
    WRAP_MIRROR = 2,
};

enum DrawFlags {
    DRAW_ZWRITE = 1,
    DRAW_BLEND = 2,
    DRAW_BLEND_EDGE = 4,
};

enum MixType {
    SH_MT_NONE            = 0,
    SH_MT_COLOR           = 1 << 0,
    SH_MT_COLOR_COLOR     = 1 << 1,
    SH_MT_TEXTURE         = 1 << 2,
    SH_MT_TEXTURE_COLOR   = 1 << 3,
    SH_MT_TEXTURE_TEXTURE = 1 << 4,
};

typedef union Vector2 { 
    struct { float x, y; };
    struct { float u, v; };
} Vector2;

typedef union Vector3 {
    struct { float r, g, b; };
    struct { float x, y, z; };
    Vector2 xy;
    float v[3];
} Vector3;

typedef union Vector4 {
    struct { float r, g, b, a; };
    struct { float x, y, z, w; };
    Vector2 xy;
    Vector3 xyz;
    float v[4];
} Vector4;

typedef union Color4 {
    struct { uint8_t r, g, b, a; };
    uint32_t c;
} Color4;

struct Tri {
    float *v0;
    float *v1;
    float *v2;
};

struct Texture;

// texture sampling function: takes integer u,v and wraps/clamps it, samples texture, returns color
typedef Color4 (*sample_fn_t)(const struct Texture * const, const int, const int);
// pixel drawing function: does blending, zwriting, alpha edge checking or whatever else, then plots pixel
typedef void (*draw_fn_t)(const int idx, uint16_t uz, const Color4 src);
// color combiner: takes float vertex properties and obtains final fragment color from them
typedef Color4 (*combine_fn_t)(const float z, const float *props);
// rasterizer: walks the triangle and interpolates a fixed amount of vertex properties
typedef void (*rast_fn_t)(const struct Tri tri);

struct ShaderProgram {
    uint32_t shader_id;
    struct CCFeatures cc;
    enum MixType mix;
    uint32_t draw_flags;
    int num_props;
    combine_fn_t combine;
    rast_fn_t rast;
};

struct Texture {
    int w, h;           // size
    float fw, fh;       // float size because float conversion bad
    int wrap_w, wrap_h; // size - 1 for wrapping
    uint8_t wrap_mode_x, wrap_mode_y; // WRAP_* per axis, resolved once (avoids indirect call per fetch)
    bool filter;        // linear filter
    uint32_t addr;      // offset into texcache
    sample_fn_t sample; // sampling function (does wrapping/clamping)
};

struct Viewport {
    int x, y, w, h; // rect
    float cx, cy;   // center
    float hw, hh;   // half size
    // float zn, zf, cz, hz; // FIXME: ztrick
};

struct ClipRect {
    int x0, y0; // top left
    int x1, y1; // bottom right
};


#if !(defined(DIRECT_SDL) && defined(SDL_SURFACE))
	#ifdef CONVERT
	uint16_t *gfx_output SAVESTATE_EXCLUDE;
	#else
	uint32_t *gfx_output SAVESTATE_EXCLUDE;
	#endif
#endif

// this is set in the drawing functions
static draw_fn_t draw_fn;

static struct ShaderProgram shader_program_pool[64];
static uint8_t shader_program_pool_size;
static struct ShaderProgram *cur_shader = NULL;

static struct Texture *cur_tex[2]; // currently selected textures for both tiles
static struct Texture tex_hdr[MAX_TEXTURES];
static uint32_t tex_num = 0; // amount of textures in cache
static int cur_tmu = 0; // select tile (used only for uploading)

// texture cache: linearly stores RGBA data of every cached texture
uint8_t *texcache;
static uint32_t texcache_addr; // current offset into cache
uint32_t texcache_size; // cache capacity

static bool do_blend; // fragment blending toggle
static bool do_clip;  // scissor toggle

static struct ClipRect r_clip;
static struct Viewport r_view;

static Color4 fog_color; // this is set by set_fog_color() calls from gfx_pc

static bool z_test;        // whether to perform depth testing
static bool z_write;       // whether to write into the Z buffer
static float z_offset;     // offset for decal mode
static uint16_t *z_buffer SAVESTATE_EXCLUDE;

static int scr_width SAVESTATE_EXCLUDE;
static int scr_height SAVESTATE_EXCLUDE;
static int scr_size SAVESTATE_EXCLUDE; // scr_width * scr_height

// OPTIMIZATION (armv7/mips32r2): lerp_tab (131 KiB) and mult_tab (64 KiB)
// were larger than L1 caches (16-32 KiB): every lookup was a near-certain
// L1 miss. Replaced by pure integer arithmetic (3-4 mla/madd instructions).
// dither kernel for unreal texture filtering
static const Vector2 dither_tab[2][2] = {
    { {{ 0.25f, 0.00f }}, {{ 0.50f, 0.75f }} },
    { {{ 0.75f, 0.50f }}, {{ 0.00f, 0.25f }} },
};

/* math shit */

static inline float fclamp01(const float v) {
    return (v < 0.f) ? 0.f : (v > 1.f) ? 1.f : v;
}

static inline uint16_t u16clamp(const int v) {
    return (v < 0) ? (uint16_t)0 : (v > 0xFFFF) ? (uint16_t)0xFFFF : (uint16_t)v;
}

static inline int iwrap0w(const int x, const int wrap) {
    return x & wrap;
}

static inline int iclamp0w(const int x, const int wrap) {
    return (x < 0) ? 0 : (x > wrap) ? wrap : x;
}

static inline int imirror0w(const int x, const int wrap) {
    return iclamp0w(abs(x), wrap); // NOTE: this is not a universal solution
}

static inline float flerp(const float v0, const float v1, const float t) {
    return v0 + t * (v1 - v0);
}

static inline bool vec2_cmp(const Vector2 v1, const Vector2 v2) {
    return (v1.y == v2.y) ? (v1.x > v2.x) : (v1.y > v2.y);
}

static inline Vector4 vec4_sub(const Vector4 *v1, const Vector4 *v2) {
    return (Vector4) {{ v1->x - v2->x, v1->y - v2->y, v1->z - v2->z, 1.f }};
}

static inline Vector4 vec4_lerp(const Vector4 *v1, const Vector4 *v2, const float t) {
    return (Vector4) {{
        flerp(v1->x, v2->x, t),
        flerp(v1->y, v2->y, t),
        flerp(v1->z, v2->z, t),
        flerp(v1->w, v2->w, t),
    }};
}

static inline Color4 rgba_modulate(const Color4 c1, const Color4 c2) {
    // integer equivalent of the old mult_tab: (x * y) / 256
    return (Color4) {{
        .r = (uint8_t)((c1.r * c2.r + 127) >> 8),
        .g = (uint8_t)((c1.g * c2.g + 127) >> 8),
        .b = (uint8_t)((c1.b * c2.b + 127) >> 8),
        .a = (uint8_t)((c1.a * c2.a + 127) >> 8),
    }};
}

static inline Color4 rgba_blend(const Color4 src, const Color4 dst, const uint8_t a) {
    const uint8_t ia = 255 - a;
    return (Color4) {{
        .r = (uint8_t)((src.r * a + dst.r * ia + 127) >> 8),
        .g = (uint8_t)((src.g * a + dst.g * ia + 127) >> 8),
        .b = (uint8_t)((src.b * a + dst.b * ia + 127) >> 8),
        .a = dst.a,
    }};
}

static inline Color4 rgba_lerp(const Color4 c1, const Color4 c2, const uint8_t t) {
    // integer equivalent of the old lerp_tab: c1 + t * (c2 - c1)
    return (Color4) {{
        .r = (uint8_t)((c2.r * t + c1.r * (255 - t) + 127) >> 8),
        .g = (uint8_t)((c2.g * t + c1.g * (255 - t) + 127) >> 8),
        .b = (uint8_t)((c2.b * t + c1.b * (255 - t) + 127) >> 8),
        .a = (uint8_t)((c2.a * t + c1.a * (255 - t) + 127) >> 8),
    }};
}

static inline int imin(const int a, const int b) {
    return (a < b) ? a : b;
}

static inline int imax(const int a, const int b) {
    return (a > b) ? a : b;
}

static inline void viewport_transform(Vector4 *v) {
    // gfx_pc.c with ENABLE_SOFTRAST defined will feed us with everything already pre-multiplied by inverse of w
    v->x = v->x * r_view.hw + r_view.cx + 0.5f;
    v->y = v->y * r_view.hh + r_view.cy + 0.5f;
    // v->w is also already 1.f / v->w
}

/* texture sampling functions */

static inline Color4 tex_get(const struct Texture * const tex, const int x, const int y) {
    return (Color4) { .c = ((const uint32_t *)(texcache + tex->addr))[y * tex->w + x] };
}

static Color4 tex_sample_nearest_rr(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, iwrap0w(x, tex->wrap_w), iwrap0w(y, tex->wrap_h));
}

static Color4 tex_sample_nearest_rc(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, iwrap0w(x, tex->wrap_w), iclamp0w(y, tex->wrap_h));
}

static Color4 tex_sample_nearest_rm(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, iwrap0w(x, tex->wrap_w), imirror0w(y, tex->wrap_h));
}

static Color4 tex_sample_nearest_cc(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, iclamp0w(x, tex->wrap_w), iclamp0w(y, tex->wrap_h));
}

static Color4 tex_sample_nearest_cr(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, iclamp0w(x, tex->wrap_w), iwrap0w(y, tex->wrap_h));
}

static Color4 tex_sample_nearest_cm(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, iclamp0w(x, tex->wrap_w), imirror0w(y, tex->wrap_h));
}

static Color4 tex_sample_nearest_mm(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, imirror0w(x, tex->wrap_w), imirror0w(y, tex->wrap_h));
}

static Color4 tex_sample_nearest_mc(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, imirror0w(x, tex->wrap_w), iclamp0w(y, tex->wrap_h));
}

static Color4 tex_sample_nearest_mr(const struct Texture * const tex, const int x, const int y) {
    return tex_get(tex, imirror0w(x, tex->wrap_w), iwrap0w(y, tex->wrap_h));
}

static inline Color4 tex_sample_linear(const struct Texture * const tex, const float u, const float v, const Vector2 d) {
    const int x = d.u + u * tex->fw;
    const int y = d.v + v * tex->fh;
    return tex->sample(tex, x, y);
}

static inline Color4 tex_sample_nearest(const struct Texture * const tex, const float u, const float v) {
    const int x = u * tex->fw;
    const int y = v * tex->fh;
    return tex->sample(tex, x, y);
}

/* =====================================================================
 * OPTIMIZATION + texture smoothing: low-cost integer bilinear filter.
 * FIX v3: gfx_pc.c ALREADY adds the +0.5-texel center offset to u,v when
 * linear filtering is active (N64 convention). Therefore the sample
 * point is exactly (u * fw) with NO extra -0.5 subtraction.
 * Fixes kept:
 *  - proper floor() instead of truncation (u*fw < 0 happens in repeat)
 *  - all taps (base and neighbors) go through tex->sample, so the true
 *    wrap mode applies: REPEAT tiles seamlessly, CLAMP/MIRROR blend
 *    with the edge texel, exactly like the N64 hardware.
 * Only used when the game requests linear filtering (configFiltering),
 * otherwise falls back to nearest at no extra cost.
 * ==================================================================== */
static inline Color4 tex_sample_bilinear_f(const struct Texture * const tex, const float u, const float v) {
    const float xf = u * tex->fw;
    const float yf = v * tex->fh;
    int x0 = (int)xf;
    int y0 = (int)yf;
    if (xf < (float)x0) --x0;
    if (yf < (float)y0) --y0;
    int fx = (int)((xf - (float)x0) * 256.f);
    int fy = (int)((yf - (float)y0) * 256.f);
    if (fx > 255) fx = 255;
    if (fy > 255) fy = 255;

    const Color4 d00 = tex->sample(tex, x0,     y0);
    const Color4 d10 = tex->sample(tex, x0 + 1, y0);
    const Color4 d01 = tex->sample(tex, x0,     y0 + 1);
    const Color4 d11 = tex->sample(tex, x0 + 1, y0 + 1);

    Color4 o;
    o.r = (uint8_t)((((d00.r * (256 - fx) + d10.r * fx) >> 8) * (256 - fy)
                   + ((d01.r * (256 - fx) + d11.r * fx) >> 8) * fy) >> 8);
    o.g = (uint8_t)((((d00.g * (256 - fx) + d10.g * fx) >> 8) * (256 - fy)
                   + ((d01.g * (256 - fx) + d11.g * fx) >> 8) * fy) >> 8);
    o.b = (uint8_t)((((d00.b * (256 - fx) + d10.b * fx) >> 8) * (256 - fy)
                   + ((d01.b * (256 - fx) + d11.b * fx) >> 8) * fy) >> 8);
    o.a = (uint8_t)((((d00.a * (256 - fx) + d10.a * fx) >> 8) * (256 - fy)
                   + ((d01.a * (256 - fx) + d11.a * fx) >> 8) * fy) >> 8);
    return o;
}

static inline Color4 tex_sample_filtered(const struct Texture * const tex, const float u, const float v) {
    return tex->filter ? tex_sample_bilinear_f(tex, u, v) : tex_sample_nearest(tex, u, v);
}

/* color combiners */

#define tex_sample tex_sample_filtered

static Color4 combine_rgb(const float z, const float *props) {
    return (Color4) {{ .r = props[0] * z, .g = props[1] * z, .b = props[2] * z, .a = 0xFF }};
}

static Color4 combine_rgba(const float z, const float *props) {
    return (Color4) {{ .r = props[0] * z, .g = props[1] * z, .b = props[2] * z, .a = props[3] * z }};
}

static Color4 combine_fog_rgb(const float z, const float *props) {
    const uint8_t fog = props[0] * z;
    const Color4 c = (Color4) {{ .r = props[1] * z, .g = props[2] * z, .b = props[3] * z, .a = 0xFF }};
    return rgba_blend(fog_color, c, fog);
}

static Color4 combine_fog_rgba(const float z, const float *props) {
    const uint8_t fog = props[0] * z;
    const Color4 c = (Color4) {{ .r = props[1] * z, .g = props[2] * z, .b = props[3] * z, .a = props[4] * z }};
    // FIX(v9.2): fog only applies to RGB. Blending alpha toward
    // fog_color.a (0xFF) made transparent texels/vertices pass the
    // edge test in heavy fog (opaque gray squares on fences etc).
    const Color4 out = rgba_blend(fog_color, c, fog);
    return (Color4) {{ .r = out.r, .g = out.g, .b = out.b, .a = c.a }};
}

static Color4 combine_rgba_rgba(const float z, const float *props) {
    const Color4 ca = (Color4) {{ .r = props[0] * z, .g = props[1] * z, .b = props[2] * z, .a = props[3] * z }};
    const Color4 cb = (Color4) {{ .r = props[4] * z, .g = props[5] * z, .b = props[6] * z, .a = props[7] * z }};
    return rgba_modulate(ca, cb);
}

static Color4 combine_tex(const float z, const float *props) {
    return tex_sample(cur_tex[0], props[0] * z, props[1] * z);
}

static Color4 combine_tex_fog(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const uint8_t fog = props[2] * z;
    // FIX(v9.2): keep the texel alpha: this combiner also serves
    // texture-edge shaders (SH_MT_TEXTURE has no alpha split), and a
    // fog-blended alpha let transparent texels pass the edge test.
    const Color4 out = rgba_blend(fog_color, tc, fog);
    return (Color4) {{ .r = out.r, .g = out.g, .b = out.b, .a = tc.a }};
}

static Color4 combine_tex_rgb(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const Color4 cc = (Color4) {{ .r = props[2] * z, .g = props[3] * z, .b = props[4] * z, .a = 0xFF }};
    return rgba_modulate(tc, cc);
}

static Color4 combine_tex_fog_rgb(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const uint8_t fog = props[2] * z;
    const Color4 cc = (Color4) {{ .r = props[3] * z, .g = props[4] * z, .b = props[5] * z, .a = 0xFF }};
    return rgba_blend(fog_color, rgba_modulate(tc, cc), fog);
}

static Color4 combine_tex_rgb_decal(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const Color4 cc = (Color4) {{ .r = props[2] * z, .g = props[3] * z, .b = props[4] * z, .a = 0xFF }};
    return rgba_blend(tc, cc, tc.a);
}

static Color4 combine_tex_rgba(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const Color4 cc = (Color4) {{ .r = props[2] * z, .g = props[3] * z, .b = props[4] * z, .a = props[5] * z }};
    return rgba_modulate(tc, cc);
}

static Color4 combine_tex_rgba_texa(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const Color4 cc = (Color4) {{ .r = props[2] * z, .g = props[3] * z, .b = props[4] * z, .a = 0xFF }};
    return rgba_modulate(tc, cc);
}

static Color4 combine_tex_fog_rgba(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const uint8_t fog = props[2] * z;
    const Color4 cc = (Color4) {{ .r = props[3] * z, .g = props[4] * z, .b = props[5] * z, .a = props[6] * z }};
    // FIX(v9.2): fog only applies to RGB (see combine_fog_rgba)
    const Color4 mod = rgba_modulate(tc, cc);
    const Color4 out = rgba_blend(fog_color, mod, fog);
    return (Color4) {{ .r = out.r, .g = out.g, .b = out.b, .a = mod.a }};
}

static Color4 combine_tex_rgba_decal(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const Color4 cc = (Color4) {{ .r = props[2] * z, .g = props[3] * z, .b = props[4] * z, .a = props[5] * z }};
    return rgba_blend(tc, cc, tc.a);
}

static Color4 combine_tex_rgba_decal_texa(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const Color4 cc = (Color4) {{ .r = props[2] * z, .g = props[3] * z, .b = props[4] * z, .a = props[5] * z }};
    Color4 out = rgba_blend(tc, cc, tc.a);
    out.a = (uint8_t)((tc.a * cc.a + 127) >> 8);
    return out;
}

/* FIX(v9.3): fog variants of the decal combiners. Enabling fog sets
 * SHADER_OPT_FOG in the shader id, so the hardcoded decal ids used to
 * stop matching when fog was on: decal geometry (fences...) fell through
 * to combine_tex_fog_rgba, which MODULATES the texel by the (grayish)
 * vertex color instead of using the texel directly -> solid gray
 * fences whenever fog was enabled. The decal ids are now compared
 * with the runtime-toggled option bits (fog/noise) masked off, and
 * these variants keep decal semantics with fog applied to RGB only
 * (never alpha, see FIX(v9.2)).
 * Layout (fog adds one prop right after u,v): [0]=u [1]=v [2]=fog
 * then colors. */
static Color4 combine_tex_fog_rgb_decal(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const uint8_t fog = props[2] * z;
    const Color4 cc = (Color4) {{ .r = props[3] * z, .g = props[4] * z, .b = props[5] * z, .a = 0xFF }};
    const Color4 base = rgba_blend(tc, cc, tc.a); // base.a == 0xFF here
    const Color4 out = rgba_blend(fog_color, base, fog);
    return (Color4) {{ .r = out.r, .g = out.g, .b = out.b, .a = base.a }};
}

static Color4 combine_tex_fog_rgba_decal(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const uint8_t fog = props[2] * z;
    const Color4 cc = (Color4) {{ .r = props[3] * z, .g = props[4] * z, .b = props[5] * z, .a = props[6] * z }};
    const Color4 base = rgba_blend(tc, cc, tc.a); // base.a == cc.a
    const Color4 out = rgba_blend(fog_color, base, fog);
    return (Color4) {{ .r = out.r, .g = out.g, .b = out.b, .a = base.a }};
}

static Color4 combine_tex_fog_rgba_decal_texa(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const uint8_t fog = props[2] * z;
    const Color4 cc = (Color4) {{ .r = props[3] * z, .g = props[4] * z, .b = props[5] * z, .a = props[6] * z }};
    const Color4 base = rgba_blend(tc, cc, tc.a);
    const Color4 out = rgba_blend(fog_color, base, fog);
    return (Color4) {{ .r = out.r, .g = out.g, .b = out.b, .a = (uint8_t)((tc.a * cc.a + 127) >> 8) }};
}

static Color4 combine_tex_rgb_rgb(const float z, const float *props) {
    const Color4 tc = tex_sample(cur_tex[0], props[0] * z, props[1] * z);
    const Color4 cc1 = (Color4) {{ .r = props[2] * z, .g = props[3] * z, .b = props[4] * z, 0xFF }};
    const Color4 cc2 = (Color4) {{ .r = props[5] * z, .g = props[6] * z, .b = props[7] * z, 0xFF }};
    return rgba_lerp(cc2, cc1, tc.r);
}

static Color4 combine_tex_tex_rgba(const float z, const float *props) {
    const float u = props[0] * z;
    const float v = props[1] * z;
    const Color4 tc1 = tex_sample(cur_tex[0], u, v);
    const Color4 tc2 = tex_sample(cur_tex[1], u, v);
    const uint8_t r = props[2] * z;
    return rgba_lerp(tc1, tc2, r);
}

/* fragment plotters */
#ifdef NOREVERSE
#define Color32Reverse(x) (x)
#else
#ifdef CONVERT
// R G B A
// R G B
static inline uint16_t Color32Reverse(uint32_t x){
    uint8_t red   = ((x >> 0)  & 0xFF);
    uint8_t green = ((x >> 8)  & 0xFF);
    uint8_t blue  = ((x >> 16)  & 0xFF);
    uint16_t b = (blue >> 3) & 0x1f;
    uint16_t g = ((green >> 2) & 0x3f) << 5;
    uint16_t r = ((red >> 3) & 0x1f) << 11;
    return (uint16_t) (r | g | b);
   /* uint32_t s = __builtin_bswap32(x) >> 8 | 0xff000000;*/
	//unsigned alpha = s >> 27; /* downscale alpha to 5 bits */
	/* FIXME: Here we special-case opaque alpha since the
	compositioning used (>>8 instead of /255) doesn't handle
	it correctly. Also special-case alpha=0 for speed?
	Benchmark this! */
	/*if(alpha == (SDL_ALPHA_OPAQUE >> 3)) return (uint16_t) ((s >> 8 & 0xf800) + (s >> 5 & 0x7e0) + (s >> 3 & 0x1f));
	return s = ((s & 0xfc00) << 11) + (s >> 8 & 0xf800) + (s >> 3 & 0x1f);*/
    
    //return (uint16_t) ((s >> 8 & 0xf800) + (s >> 5 & 0x7e0) + (s >> 3 & 0x1f));
    
	/*extern SDL_Surface* sdl_screen;
	uint8_t a		= ((x >> 24) & 0xFF);
    uint8_t red		= ((x >> 0)  & 0xFF);
    uint8_t green	= ((x >> 8 ) & 0xFF);
    uint8_t blue	= ((x >> 16) & 0xFF);
    red = red >> 3;
    green = green >> 2;
    blue = blue >> 3;
	return SDL_MapRGBA(sdl_screen->format, red, green, blue, a);*/
}
#else
static inline uint32_t Color32Reverse(uint32_t x)
{
    return __builtin_bswap32(x) >> 8 | 0xff000000;
}
#endif
#endif

static void draw_pixel(const int idx, UNUSED const uint16_t z, Color4 src) {
    gfx_output[idx] = Color32Reverse(src.c);
}

static void draw_pixel_zwrite(const int idx, const uint16_t z, Color4 src) {
    gfx_output[idx] = Color32Reverse(src.c);
    z_buffer[idx] = z;
}

// FIX(v8): read the framebuffer destination for blending. The old code
// passed the raw pixel to Color32Reverse(), which under CONVERT
// interprets the 16bpp RGB565 value as 32bpp (a<<24|b<<16|g<<8|r):
// every blended pixel on the generic path (shadows, fades, fog
// geometry, text antialiasing) read scrambled r/g/b channels.
static inline Color4 gfx_read_dst(const int idx) {
#ifdef CONVERT
    const uint16_t d = gfx_output[idx];
    const int r5 = (d >> 11) & 0x1F, g6 = (d >> 5) & 0x3F, b5 = d & 0x1F;
    return (Color4) {{
        .r = (uint8_t)((r5 << 3) | (r5 >> 2)),
        .g = (uint8_t)((g6 << 2) | (g6 >> 4)),
        .b = (uint8_t)((b5 << 3) | (b5 >> 2)),
        .a = 0xFF,
    }};
#else
    return (Color4) { .c = Color32Reverse(gfx_output[idx]) };
#endif
}

static void draw_pixel_blend(const int idx, UNUSED const uint16_t z, Color4 src) {
    const uint8_t a = src.a;
    const uint8_t ia = 255 - a;
    const Color4 dst = gfx_read_dst(idx);
    src.r = (uint8_t)((src.r * a + dst.r * ia + 127) >> 8);
    src.g = (uint8_t)((src.g * a + dst.g * ia + 127) >> 8);
    src.b = (uint8_t)((src.b * a + dst.b * ia + 127) >> 8);
    gfx_output[idx] = Color32Reverse(src.c);
}

static void draw_pixel_blend_zwrite(const int idx, const uint16_t z, Color4 src) {
    const uint8_t a = src.a;
    const uint8_t ia = 255 - a;
    const Color4 dst = gfx_read_dst(idx);
    src.r = (uint8_t)((src.r * a + dst.r * ia + 127) >> 8);
    src.g = (uint8_t)((src.g * a + dst.g * ia + 127) >> 8);
    src.b = (uint8_t)((src.b * a + dst.b * ia + 127) >> 8);
    gfx_output[idx] = Color32Reverse(src.c);
    z_buffer[idx] = z;
}

static void draw_pixel_blend_edge(const int idx, UNUSED const uint16_t z, Color4 src) {
    if (src.a > 0x80) {
        const uint8_t a = src.a;
        const uint8_t ia = 255 - a;
        const Color4 dst = gfx_read_dst(idx);
        src.r = (uint8_t)((src.r * a + dst.r * ia + 127) >> 8);
        src.g = (uint8_t)((src.g * a + dst.g * ia + 127) >> 8);
        src.b = (uint8_t)((src.b * a + dst.b * ia + 127) >> 8);
        gfx_output[idx] = Color32Reverse(src.c);
    }
}

static void draw_pixel_blend_edge_zwrite(const int idx, const uint16_t z, Color4 src) {
    if (src.a > 0x80) {
        const uint8_t a = src.a;
        const uint8_t ia = 255 - a;
        const Color4 dst = gfx_read_dst(idx);
        src.r = (uint8_t)((src.r * a + dst.r * ia + 127) >> 8);
        src.g = (uint8_t)((src.g * a + dst.g * ia + 127) >> 8);
        src.b = (uint8_t)((src.b * a + dst.b * ia + 127) >> 8);
        gfx_output[idx] = Color32Reverse(src.c);
        z_buffer[idx] = z;
    }
}

/* rasterizers */

#define R_RASTERIZE_TRI_SEG(y_a, y_b, nprops) \
    register int y = y_a; \
    register int y_end = y_b; \
    register int x, x_end; \
    register int idx; \
    register float dx, w; \
    uint16_t uz; \
    /* draw triangle segment from y_a to y_b */ \
    while (y < y_end) { \
        /* do scissor clipping */ \
        x = imax(r_clip.x0, x_a); \
        x_end = imin(r_clip.x1, x_b); \
        /* do X subpixel prestepping */ \
        dx = 1.f - (x_a - x); \
        for (i = 2; i < nprops; ++i) p[i] = p_a[i] + dx * dp[i].x; \
        idx = scr_width * (scr_height - y - 1) + x; \
        /* draw scanline from current x_a to current x_b */ \
        while (x++ < x_end) { \
            uz = u16clamp(p[2] * 65535.f + z_offset); \
            if (!z_test || uz <= z_buffer[idx]) { \
                w = 1.f / p[3]; /* the combiner will multiply by w any props it needs to persp correct */ \
                draw_fn(idx, uz, cur_shader->combine(w, p + 4)); \
            } \
            for (i = 2; i < nprops; ++i) p[i] += dp[i].x; \
            ++idx; \
        } \
        /* advance scanline start and end and prop starts */ \
        x_a += dxdy_a; \
        x_b += dxdy_b; \
        for (i = 2; i < nprops; ++i) p_a[i] += dpdy_a[i]; \
        ++y; \
    }

#define R_RASTERIZE(tri, nprops) \
    const float *v0 = (float *)tri.v0; \
    const float *v1 = (float *)tri.v1; \
    const float *v2 = (float *)tri.v2; \
    const int y0i = imax(r_clip.y0, (int)v0[1]); \
    const int y1i = imax(y0i, (int)v1[1]); \
    const int y2i = imin(r_clip.y1, (int)v2[1]); \
    if ((y0i == y1i && y0i == y2i) || ((int)v0[0] == (int)v1[0] && (int)v0[0] == (int)v2[0])) \
        return; /* triangle has zero area */ \
    const Vector4 ab = (Vector4) {{ v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], v1[3] - v0[3] }}; \
    const Vector4 ac = (Vector4) {{ v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2], v2[3] - v0[3] }}; \
    const Vector2 bc = (Vector2) {{ v2[0] - v1[0], v2[1] - v1[1] }}; \
    const float denom = 1.f / (ac.x * ab.y - ab.x * ac.y); \
    const float dxdy_ab = ab.x / ab.y; /* x increment along ab */ \
    const float dxdy_ac = ac.x / ac.y; /* x increment along ac */ \
    const float dxdy_bc = bc.x / bc.y; /* x increment along bc */ \
    const bool side = dxdy_ac > dxdy_ab; /* which side the longer edge (AC) is on */ \
    const float y_pre0 = 1.f - (v0[1] - y0i); /* subpixel pre-step */ \
    float dpdy_a[nprops]; /* vertex prop increments along left edge */ \
    float p_a[nprops]; /* vertex leftmost points */ \
    float p[nprops]; /* current vertex prop values */ \
    Vector2 dp[nprops]; /* X and Y increments for vertex props */ \
    register int i; \
    /* we'll interpolate z/w (p[2]), 1/w (p[3]) and the other properties (also divided by w) */ \
    for (i = 2; i < nprops; ++i) { \
        dp[i].x = ((v2[i] - v0[i]) * ab.y - (v1[i] - v0[i]) * ac.y) * denom; \
        dp[i].y = ((v1[i] - v0[i]) * ac.x - (v2[i] - v0[i]) * ab.x) * denom; \
    } \
    if (!side) { \
        /* longer edge is on the left */ \
        const float dxdy_a = dxdy_ac; \
        /* first column of this scanline is on AC */ \
        float x_a = v0[0] + y_pre0 * dxdy_a; \
        for (i = 2; i < nprops; ++i) { \
            dpdy_a[i] = dxdy_ac * dp[i].x + dp[i].y; \
            p_a[i] = v0[i] + y_pre0 * dpdy_a[i]; \
        } \
        if (y0i < y1i) { \
            /* left is AC, right is AB */ \
            const float dxdy_b = dxdy_ab; \
            /* last column of this scanline */ \
            float x_b = v0[0] + y_pre0 * dxdy_ab; \
            R_RASTERIZE_TRI_SEG(y0i, y1i, nprops); \
        } \
        if (y1i < y2i) { \
            /* left is AC, right is BC */ \
            const float dxdy_b = dxdy_bc; \
            /* calculate prestep for vertex B */ \
            const float y_pre1 = 1.f - (v1[1] - y1i); \
            float x_b = v1[0] + y_pre1 * dxdy_bc; \
            R_RASTERIZE_TRI_SEG(y1i, y2i, nprops); \
        } \
    } else { \
        /* longer edge is on the right */ \
        const float dxdy_b = dxdy_ac; \
        /* last column of this scanline is on AC */ \
        float x_b = v0[0] + y_pre0 * dxdy_ac; \
        if (y0i < y1i) { \
            /* right is AC, left is AB */ \
            const float dxdy_a = dxdy_ab; \
            float x_a = v0[0] + y_pre0 * dxdy_a; \
            for (i = 2; i < nprops; ++i) { \
                dpdy_a[i] = dxdy_ab * dp[i].x + dp[i].y; \
                p_a[i] = v0[i] + y_pre0 * dpdy_a[i]; \
            } \
            R_RASTERIZE_TRI_SEG(y0i, y1i, nprops); \
        } \
        if (y1i < y2i) { \
            /* right is AC, left is BC */ \
            const float y_pre1 = 1.f - (v1[1] - y1i); \
            const float dxdy_a = dxdy_bc; \
            float x_a = v1[0] + y_pre1 * dxdy_a; \
            for (i = 2; i < nprops; ++i) { \
                dpdy_a[i] = dxdy_bc * dp[i].x + dp[i].y; \
                p_a[i] = v1[i] + y_pre1 * dpdy_a[i]; \
            } \
            R_RASTERIZE_TRI_SEG(y1i, y2i, nprops); \
        } \
    }

// define a bunch of rasterizers/interpolators for known property counts
// nprops includes XYZW

#define DEFINE_RAST_FUNC(nprops) \
    static void rast_fn_ ## nprops (const struct Tri tri) { R_RASTERIZE(tri, nprops); }

#define GET_RAST_FUNC(nprops) rast_fn_ ## nprops

DEFINE_RAST_FUNC(6)
DEFINE_RAST_FUNC(7)
DEFINE_RAST_FUNC(8)
DEFINE_RAST_FUNC(9)
DEFINE_RAST_FUNC(10)
DEFINE_RAST_FUNC(11)
DEFINE_RAST_FUNC(12)
DEFINE_RAST_FUNC(13)
DEFINE_RAST_FUNC(14)

/* =====================================================================
 * OPTIMIZATION: fused fast-path scanlines for the 4 dominant shaders:
 *   combine_tex_rgb  (opaque textured)          nprops 7? no: 9 (u,v,rgb)
 *   combine_tex_rgba (textured + alpha blending) nprops 10 (u,v,r,g,b,a)
 *   combine_rgba     (vertex color + alpha)      nprops 8 (r,g,b,a)
 *   combine_rgb      (vertex color, opaque)      nprops 7 (r,g,b)
 * For these paths only:
 *   - 1 float division per pixel   -> 1 division per OPT_STEP pixels
 *   - 2 indirect calls per pixel   -> 0 (combiner and plotter inlined)
 *   - float interpolation per pixel -> 16.16 / 8.8 integer stepping
 * Blend / texture-edge / z-write are resolved from cur_shader->draw_flags
 * and the z_test / z_write globals (same semantics as draw_pixel_*).
 * Bilinear (v3): sample point is (u*fw) directly (gfx_pc.c already adds
 * the N64 +0.5-texel center offset); all taps use the true wrap mode.
 * CONVERT (RGB565) builds use the same paths via opt_pix_t / opt_put /
 * opt_get: framebuffer bandwidth is halved with the same fast paths.
 * ==================================================================== */

#ifndef CONVERT
typedef uint32_t opt_pix_t;
#else
typedef uint16_t opt_pix_t; // RGB565, halves framebuffer bandwidth
#endif

#define OPT_STEP 8   /* 4 on MIPS32r2 if banding appears, 16 for more speed */

// RGB565 pack/unpack for the CONVERT build (matches Color32Reverse(CONVERT):
// input r in bits 0-7, g in 8-15, b in 16-23 -> r<<11 | g<<5 | b)
static inline uint16_t opt_pack565(const int r, const int g, const int b) {
    return (uint16_t)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
}
static inline void opt_unpack565(const uint16_t d, int *r, int *g, int *b) {
    // FIX(v8): replicate high bits for full 0-255 range (the old plain
    // shifts mapped white to (248,252,248) -> greenish tint accumulating
    // over successive blends: visible on shadows and fades)
    const int r5 = (d >> 11) & 0x1F, g6 = (d >> 5) & 0x3F, b5 = d & 0x1F;
    *r = (r5 << 3) | (r5 >> 2);
    *g = (g6 << 2) | (g6 >> 4);
    *b = (b5 << 3) | (b5 >> 2);
}

// pixel write/read valid in both 32bpp and CONVERT (RGB565) builds
static inline void opt_put(opt_pix_t *dst, const int r, const int g, const int b) {
#ifndef CONVERT
    // FIX(v9.1): must match Color32Reverse() = bswap32(x) >> 8 | 0xFF000000,
    // which puts BLUE in the low byte and RED in bits 16-23 of the 32bpp
    // framebuffer. The previous (b << 16 | r) layout swapped R and B on
    // every fast-path pixel (bug was hidden while the routing constants
    // were wrong and the fast paths never ran).
    *dst = 0xFF000000u | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
#else
    *dst = opt_pack565(r, g, b);
#endif
}
static inline void opt_get(const opt_pix_t *src, int *r, int *g, int *b) {
#ifndef CONVERT
    // FIX(v9.1): inverse of opt_put (was swapped like opt_put)
    *r = (*src >> 16) & 0xFF;
    *g = (*src >> 8) & 0xFF;
    *b = *src & 0xFF;
#else
    opt_unpack565(*src, r, g, b);
#endif
}

static inline int opt_wrap(const int c, const int wrap, const uint8_t mode) {
    if (mode == WRAP_REPEAT) return c & wrap;
    if (mode == WRAP_CLAMP) return iclamp0w(c, wrap);
    return imirror0w(c, wrap);
}

/* --- opaque textured (tex_rgb): v3 path, unchanged --- */
static void opt_scan_tex_rgb(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    const int fw16 = (int)(tex->fw * 65536.0f);
    const int fh16 = (int)(tex->fh * 65536.0f);
    const int zoff = (int)z_offset;
    const uint8_t wm_x = tex->wrap_mode_x, wm_y = tex->wrap_mode_y;
    const int tw = tex->wrap_w, th = tex->wrap_h;
    const uint32_t *tpix = (const uint32_t *)(texcache + tex->addr);
    const bool bil = tex->filter;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int uu = (int)(p[4] * w0 * fw16);
        int vv = (int)(p[5] * w0 * fh16);
        int rr = (int)(p[6] * w0);
        int gg = (int)(p[7] * w0);
        int bb = (int)(p[8] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[9];
        for (int i = 2; i < 9; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int uu1 = (int)(q[4] * w1 * fw16);
        const int vv1 = (int)(q[5] * w1 * fh16);
        const int rr1 = (int)(q[6] * w1);
        const int gg1 = (int)(q[7] * w1);
        const int bb1 = (int)(q[8] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int du = (uu1 - uu) / cnt;
        const int dv = (vv1 - vv) / cnt;
        const int dr = (rr1 - rr) / cnt;
        const int dg = (gg1 - gg) / cnt;
        const int db = (bb1 - bb) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            uint32_t tc;
            if (bil) {
                const int x0 = uu >> 16, y0 = vv >> 16;
                const int fx = (uu >> 8) & 0xFF, fy = (vv >> 8) & 0xFF;
                const int xb = opt_wrap(x0, tw, wm_x), yb = opt_wrap(y0, th, wm_y);
                const int xn = opt_wrap(x0 + 1, tw, wm_x), yn = opt_wrap(y0 + 1, th, wm_y);
                const uint32_t c00 = tpix[yb * tex->w + xb];
                const uint32_t c10 = tpix[yb * tex->w + xn];
                const uint32_t c01 = tpix[yn * tex->w + xb];
                const uint32_t c11 = tpix[yn * tex->w + xn];
                uint32_t out = 0xFF000000u;
                for (uint32_t ch = 0; ch < 3; ++ch) {
                    const uint32_t sh = ch * 8;
                    const int t = (int)((c00 >> sh) & 0xFF) * (256 - fx) + (int)((c10 >> sh) & 0xFF) * fx;
                    const int b = (int)((c01 >> sh) & 0xFF) * (256 - fx) + (int)((c11 >> sh) & 0xFF) * fx;
                    out |= ((uint32_t)((((t >> 8) * (256 - fy) + (b >> 8) * fy) >> 8)) << sh);
                }
                tc = out;
            } else {
                tc = tpix[opt_wrap(vv >> 16, th, wm_y) * tex->w + opt_wrap(uu >> 16, tw, wm_x)];
            }
            if (!z_test || (uint16_t)zz <= *zb) {
                const int tr = (tc & 0xFF) * rr;
                const int tg = ((tc >> 8) & 0xFF) * gg;
                const int tb = ((tc >> 16) & 0xFF) * bb;
                opt_put(dst, (tr + 127) >> 8, (tg + 127) >> 8, (tb + 127) >> 8);
                if (z_write) *zb = (uint16_t)zz;
            }
            ++dst; ++zb;
            uu += du; vv += dv;
            rr += dr; gg += dg; bb += db;
            zz += dz;
        }
        for (int i = 2; i < 9; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- textured + alpha (tex_rgba): blend / texture_edge variants --- */
static void opt_scan_tex_rgba(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    const int fw16 = (int)(tex->fw * 65536.0f);
    const int fh16 = (int)(tex->fh * 65536.0f);
    const int zoff = (int)z_offset;
    const uint8_t wm_x = tex->wrap_mode_x, wm_y = tex->wrap_mode_y;
    const int tw = tex->wrap_w, th = tex->wrap_h;
    const uint32_t *tpix = (const uint32_t *)(texcache + tex->addr);
    const bool bil = tex->filter;
    // draw mode resolved once per scanline (matches draw_pixel_* family).
    // FIX(v9.2): texture-edge shaders carry DRAW_BLEND_EDGE *without*
    // DRAW_BLEND (see create_and_load_new_shader), so testing DRAW_BLEND
    // alone made every edge-mode pixel take the opaque-write branch:
    // coins / shadows / bubbles were drawn as solid squares.
    const uint32_t dfl = cur_shader->draw_flags;
    const bool do_blend = (dfl & (DRAW_BLEND | DRAW_BLEND_EDGE)) != 0;
    const bool do_edge  = (dfl & DRAW_BLEND_EDGE) != 0;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int uu = (int)(p[4] * w0 * fw16);
        int vv = (int)(p[5] * w0 * fh16);
        int rr = (int)(p[6] * w0);
        int gg = (int)(p[7] * w0);
        int bb = (int)(p[8] * w0);
        int aa = (int)(p[9] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[10];
        for (int i = 2; i < 10; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int uu1 = (int)(q[4] * w1 * fw16);
        const int vv1 = (int)(q[5] * w1 * fh16);
        const int rr1 = (int)(q[6] * w1);
        const int gg1 = (int)(q[7] * w1);
        const int bb1 = (int)(q[8] * w1);
        const int aa1 = (int)(q[9] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int du = (uu1 - uu) / cnt;
        const int dv = (vv1 - vv) / cnt;
        const int dr = (rr1 - rr) / cnt;
        const int dg = (gg1 - gg) / cnt;
        const int db = (bb1 - bb) / cnt;
        const int da = (aa1 - aa) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            /* OPTIMIZATION (v11): alpha-first early-out for blend/edge
             * shaders (masks, overlays cover the whole screen but are
             * mostly transparent). Compute the alpha channel first
             * (1-channel bilinear), reject transparent pixels, and only
             * pay the 3-channel bilinear + modulate + dst blend for
             * pixels that are actually written. Same visible result:
             * blend with sa==0 writes dst back unchanged, edge below
             * threshold writes nothing (original draw_pixel_* behavior).
             */
            uint32_t c00, c10, c01, c11;
            int fx, fy;
            int ta;
            if (bil) {
                const int x0 = uu >> 16, y0 = vv >> 16;
                fx = (uu >> 8) & 0xFF; fy = (vv >> 8) & 0xFF;
                const int xb = opt_wrap(x0, tw, wm_x), yb = opt_wrap(y0, th, wm_y);
                const int xn = opt_wrap(x0 + 1, tw, wm_x), yn = opt_wrap(y0 + 1, th, wm_y);
                c00 = tpix[yb * tex->w + xb];
                c10 = tpix[yb * tex->w + xn];
                c01 = tpix[yn * tex->w + xb];
                c11 = tpix[yn * tex->w + xn];
                const int t = (int)((c00 >> 24) & 0xFF) * (256 - fx) + (int)((c10 >> 24) & 0xFF) * fx;
                const int b = (int)((c01 >> 24) & 0xFF) * (256 - fx) + (int)((c11 >> 24) & 0xFF) * fx;
                ta = (((((t >> 8) * (256 - fy) + (b >> 8) * fy) >> 8)) * aa + 127) >> 8;
            } else {
                fx = 0; fy = 0;
                c00 = tpix[opt_wrap(vv >> 16, th, wm_y) * tex->w + opt_wrap(uu >> 16, tw, wm_x)];
                c10 = c01 = c11 = c00;
                ta = ((int)((c00 >> 24) & 0xFF) * aa + 127) >> 8;
            }
            if (!z_test || (uint16_t)zz <= *zb) {
                if (do_edge && ta <= 0x80) {
                    // below the edge threshold: nothing written at all
                } else if (!do_blend) {
                    // opaque write (draw_pixel / draw_pixel_zwrite)
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    opt_put(dst, (tr * rr + 127) >> 8, (tg * gg + 127) >> 8, (tb * bb + 127) >> 8);
                    if (z_write) *zb = (uint16_t)zz;
                } else if (ta == 0) {
                    // blend with 0 alpha: the color write is a no-op,
                    // keep only the z write of draw_pixel_blend_zwrite
                    if (z_write) *zb = (uint16_t)zz;
                } else {
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    const int sr = (tr * rr + 127) >> 8;
                    const int sg = (tg * gg + 127) >> 8;
                    const int sb = (tb * bb + 127) >> 8;
                    if (ta >= 255) {
                        opt_put(dst, sr, sg, sb);
                    } else {
                        int dr_, dg_, db_;
                        opt_get(dst, &dr_, &dg_, &db_);
                        const int ia = 255 - ta;
                        const int orr = (sr * ta + dr_ * ia + 127) >> 8;
                        const int og = (sg * ta + dg_ * ia + 127) >> 8;
                        const int ob = (sb * ta + db_ * ia + 127) >> 8;
                        opt_put(dst, orr, og, ob);
                    }
                    if (z_write) *zb = (uint16_t)zz;
                }
            }
            ++dst; ++zb;
            uu += du; vv += dv;
            rr += dr; gg += dg; bb += db; aa += da;
            zz += dz;
        }
        for (int i = 2; i < 10; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- vertex color + alpha (rgba): mist, transparent geometry --- */
static void opt_scan_rgba(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, UNUSED const struct Texture * const tex) {
    const int zoff = (int)z_offset;
    const uint32_t dfl = cur_shader->draw_flags;
    // FIX(v9.2): include DRAW_BLEND_EDGE (see opt_scan_tex_rgba)
    const bool do_blend = (dfl & (DRAW_BLEND | DRAW_BLEND_EDGE)) != 0;
    const bool do_edge  = (dfl & DRAW_BLEND_EDGE) != 0;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int rr = (int)(p[4] * w0);
        int gg = (int)(p[5] * w0);
        int bb = (int)(p[6] * w0);
        int aa = (int)(p[7] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[8];
        for (int i = 2; i < 8; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int rr1 = (int)(q[4] * w1);
        const int gg1 = (int)(q[5] * w1);
        const int bb1 = (int)(q[6] * w1);
        const int aa1 = (int)(q[7] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int dr = (rr1 - rr) / cnt;
        const int dg = (gg1 - gg) / cnt;
        const int db = (bb1 - bb) / cnt;
        const int da = (aa1 - aa) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            if (!z_test || (uint16_t)zz <= *zb) {
                if (!do_blend) {
                    opt_put(dst, rr, gg, bb);
                    if (z_write) *zb = (uint16_t)zz;
                } else if (!do_edge || aa > 0x80) {
                    if (aa >= 255) {
                        opt_put(dst, rr, gg, bb);
                    } else if (aa > 0) {
                        int dr_, dg_, db_;
                        opt_get(dst, &dr_, &dg_, &db_);
                        const int ia = 255 - aa;
                        const int orr = (rr * aa + dr_ * ia + 127) >> 8;
                        const int og = (gg * aa + dg_ * ia + 127) >> 8;
                        const int ob = (bb * aa + db_ * ia + 127) >> 8;
                        opt_put(dst, orr, og, ob);
                    }
                    if (z_write) *zb = (uint16_t)zz;
                }
            }
            ++dst; ++zb;
            rr += dr; gg += dg; bb += db; aa += da;
            zz += dz;
        }
        for (int i = 2; i < 8; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- opaque vertex color (rgb): HUD, untextured geometry --- */static void opt_scan_rgb(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, UNUSED const struct Texture * const tex) {
    const int zoff = (int)z_offset;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int rr = (int)(p[4] * w0);
        int gg = (int)(p[5] * w0);
        int bb = (int)(p[6] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[7];
        for (int i = 2; i < 7; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int rr1 = (int)(q[4] * w1);
        const int gg1 = (int)(q[5] * w1);
        const int bb1 = (int)(q[6] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int dr = (rr1 - rr) / cnt;
        const int dg = (gg1 - gg) / cnt;
        const int db = (bb1 - bb) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            if (!z_test || (uint16_t)zz <= *zb) {
                opt_put(dst, rr, gg, bb);
                if (z_write) *zb = (uint16_t)zz;
            }
            ++dst; ++zb;
            rr += dr; gg += dg; bb += db;
            zz += dz;
        }
        for (int i = 2; i < 7; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* =====================================================================
 * OPTIMIZATION (v9): fused fog fast paths. Layout (GFX_W_PREMULT, props
 * are stored divided by w, multiplied back by w0 = 1/p[3] here):
 *   combine_fog_rgb      (nprops 8):  p[4]=fog, p[5..7]=r,g,b
 *   combine_tex_fog      (nprops 7):  p[4]=u, p[5]=v, p[6]=fog
 *   combine_tex_fog_rgb  (nprops 10): p[4]=u, p[5]=v, p[6]=fog, p[7..9]=r,g,b
 * Fog factor is interpolated like any other prop (same math as the
 * generic rasterizer, no visual change), then each pixel is blended
 * toward fog_color with integer math: (fog*c + (255-fog)*fc + 127) >> 8.
 * All three shaders are opaque (draw_flags == 0), so only z_test/z_write.
 * ==================================================================== */

/* --- untextured + fog (combine_fog_rgb): distant terrain, walls --- */
static void opt_scan_fog_rgb(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, UNUSED const struct Texture * const tex) {
    const int zoff = (int)z_offset;
    const int fcr = fog_color.r, fcg = fog_color.g, fcb = fog_color.b;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int ff = (int)(p[4] * w0);
        int rr = (int)(p[5] * w0);
        int gg = (int)(p[6] * w0);
        int bb = (int)(p[7] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[8];
        for (int i = 2; i < 8; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int ff1 = (int)(q[4] * w1);
        const int rr1 = (int)(q[5] * w1);
        const int gg1 = (int)(q[6] * w1);
        const int bb1 = (int)(q[7] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int df = (ff1 - ff) / cnt;
        const int dr = (rr1 - rr) / cnt;
        const int dg = (gg1 - gg) / cnt;
        const int db = (bb1 - bb) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            if (!z_test || (uint16_t)zz <= *zb) {
                if (ff < 0) ff = 0; else if (ff > 255) ff = 255;
                const int ia = 255 - ff;
                // FIX(v9.5): the fog blend was INVERTED (ff*c + ia*fog):
                // at fog=0 pixels got pure fog_color and at fog=255 pure
                // color -> every tex_fog surface (fences, iron bars...)
                // was uniformly gray regardless of distance. Correct
                // direction, matching rgba_blend(fog_color, c, fog):
                // out = fog*fog_color + (255-fog)*color.
                opt_put(dst,
                    (ia * rr + ff * fcr + 127) >> 8,
                    (ia * gg + ff * fcg + 127) >> 8,
                    (ia * bb + ff * fcb + 127) >> 8);
                if (z_write) *zb = (uint16_t)zz;
            }
            ++dst; ++zb;
            ff += df; rr += dr; gg += dg; bb += db;
            zz += dz;
        }
        for (int i = 2; i < 8; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- textured + fog, no shade (combine_tex_fog) --- */
static void opt_scan_tex_fog(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    const int fw16 = (int)(tex->fw * 65536.0f);
    const int fh16 = (int)(tex->fh * 65536.0f);
    const int zoff = (int)z_offset;
    const uint8_t wm_x = tex->wrap_mode_x, wm_y = tex->wrap_mode_y;
    const int tw = tex->wrap_w, th = tex->wrap_h;
    const uint32_t *tpix = (const uint32_t *)(texcache + tex->addr);
    const bool bil = tex->filter;
    const int fcr = fog_color.r, fcg = fog_color.g, fcb = fog_color.b;
    // FIX(v9.2): this combiner also serves texture-edge shaders
    // (SH_MT_TEXTURE has no alpha split), resolve draw mode like the
    // other fast paths. Source alpha = texel alpha (fog only affects RGB).
    const uint32_t dfl = cur_shader->draw_flags;
    const bool do_blend = (dfl & (DRAW_BLEND | DRAW_BLEND_EDGE)) != 0;
    const bool do_edge  = (dfl & DRAW_BLEND_EDGE) != 0;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int uu = (int)(p[4] * w0 * fw16);
        int vv = (int)(p[5] * w0 * fh16);
        int ff = (int)(p[6] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[7];
        for (int i = 2; i < 7; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int uu1 = (int)(q[4] * w1 * fw16);
        const int vv1 = (int)(q[5] * w1 * fh16);
        const int ff1 = (int)(q[6] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int du = (uu1 - uu) / cnt;
        const int dv = (vv1 - vv) / cnt;
        const int df = (ff1 - ff) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            /* OPTIMIZATION (v11): alpha-first early-out (see
             * opt_scan_tex_rgba): texture-edge shaders (masks, sky edge
             * quads) are mostly transparent. Compute the alpha channel
             * first (1-channel bilinear), reject transparent pixels, and
             * only pay the 3-channel bilinear + fog blend for pixels that
             * are actually written. Source alpha = texel alpha (fog only
             * affects RGB, see v9.2). */
            uint32_t c00, c10, c01, c11;
            int fx, fy;
            int ta;
            if (bil) {
                const int x0 = uu >> 16, y0 = vv >> 16;
                fx = (uu >> 8) & 0xFF; fy = (vv >> 8) & 0xFF;
                const int xb = opt_wrap(x0, tw, wm_x), yb = opt_wrap(y0, th, wm_y);
                const int xn = opt_wrap(x0 + 1, tw, wm_x), yn = opt_wrap(y0 + 1, th, wm_y);
                c00 = tpix[yb * tex->w + xb];
                c10 = tpix[yb * tex->w + xn];
                c01 = tpix[yn * tex->w + xb];
                c11 = tpix[yn * tex->w + xn];
                const int t = (int)((c00 >> 24) & 0xFF) * (256 - fx) + (int)((c10 >> 24) & 0xFF) * fx;
                const int b = (int)((c01 >> 24) & 0xFF) * (256 - fx) + (int)((c11 >> 24) & 0xFF) * fx;
                ta = ((t >> 8) * (256 - fy) + (b >> 8) * fy) >> 8;
            } else {
                fx = 0; fy = 0;
                c00 = tpix[opt_wrap(vv >> 16, th, wm_y) * tex->w + opt_wrap(uu >> 16, tw, wm_x)];
                c10 = c01 = c11 = c00;
                ta = (int)((c00 >> 24) & 0xFF);
            }
            if (!z_test || (uint16_t)zz <= *zb) {
                if (do_edge && ta <= 0x80) {
                    // below the edge threshold: nothing written at all
                } else {
                    if (ff < 0) ff = 0; else if (ff > 255) ff = 255;
                    const int ia = 255 - ff;
                    // fogged RGB (fog applies to color only, never alpha)
                    // FIX(v9.5): fog blend direction (was inverted, see
                    // opt_scan_fog_rgb): out = fog*fog_color + (255-fog)*texel
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    const int fr = (ia * tr + ff * fcr + 127) >> 8;
                    const int fg = (ia * tg + ff * fcg + 127) >> 8;
                    const int fb = (ia * tb + ff * fcb + 127) >> 8;
                    if (!do_blend) {
                        opt_put(dst, fr, fg, fb);
                        if (z_write) *zb = (uint16_t)zz;
                    } else if (ta == 0) {
                        // blend with 0 alpha: the color write is a no-op,
                        // keep only the z write (draw_pixel_blend_zwrite)
                        if (z_write) *zb = (uint16_t)zz;
                    } else {
                        if (ta >= 255) {
                            opt_put(dst, fr, fg, fb);
                        } else {
                            int dr_, dg_, db_;
                            opt_get(dst, &dr_, &dg_, &db_);
                            const int iat = 255 - ta;
                            opt_put(dst,
                                (fr * ta + dr_ * iat + 127) >> 8,
                                (fg * ta + dg_ * iat + 127) >> 8,
                                (fb * ta + db_ * iat + 127) >> 8);
                        }
                        if (z_write) *zb = (uint16_t)zz;
                    }
                }
            }
            ++dst; ++zb;
            uu += du; vv += dv; ff += df;
            zz += dz;
        }
        for (int i = 2; i < 7; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- textured + fog + shade (combine_tex_fog_rgb): the common case --- */
static void opt_scan_tex_fog_rgb(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    const int fw16 = (int)(tex->fw * 65536.0f);
    const int fh16 = (int)(tex->fh * 65536.0f);
    const int zoff = (int)z_offset;
    const uint8_t wm_x = tex->wrap_mode_x, wm_y = tex->wrap_mode_y;
    const int tw = tex->wrap_w, th = tex->wrap_h;
    const uint32_t *tpix = (const uint32_t *)(texcache + tex->addr);
    const bool bil = tex->filter;
    const int fcr = fog_color.r, fcg = fog_color.g, fcb = fog_color.b;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int uu = (int)(p[4] * w0 * fw16);
        int vv = (int)(p[5] * w0 * fh16);
        int ff = (int)(p[6] * w0);
        int rr = (int)(p[7] * w0);
        int gg = (int)(p[8] * w0);
        int bb = (int)(p[9] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[10];
        for (int i = 2; i < 10; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int uu1 = (int)(q[4] * w1 * fw16);
        const int vv1 = (int)(q[5] * w1 * fh16);
        const int ff1 = (int)(q[6] * w1);
        const int rr1 = (int)(q[7] * w1);
        const int gg1 = (int)(q[8] * w1);
        const int bb1 = (int)(q[9] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int du = (uu1 - uu) / cnt;
        const int dv = (vv1 - vv) / cnt;
        const int df = (ff1 - ff) / cnt;
        const int dr = (rr1 - rr) / cnt;
        const int dg = (gg1 - gg) / cnt;
        const int db = (bb1 - bb) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            uint32_t tc;
            if (bil) {
                const int x0 = uu >> 16, y0 = vv >> 16;
                const int fx = (uu >> 8) & 0xFF, fy = (vv >> 8) & 0xFF;
                const int xb = opt_wrap(x0, tw, wm_x), yb = opt_wrap(y0, th, wm_y);
                const int xn = opt_wrap(x0 + 1, tw, wm_x), yn = opt_wrap(y0 + 1, th, wm_y);
                const uint32_t c00 = tpix[yb * tex->w + xb];
                const uint32_t c10 = tpix[yb * tex->w + xn];
                const uint32_t c01 = tpix[yn * tex->w + xb];
                const uint32_t c11 = tpix[yn * tex->w + xn];
                uint32_t out = 0xFF000000u;
                for (uint32_t ch = 0; ch < 3; ++ch) {
                    const uint32_t sh = ch * 8;
                    const int t = (int)((c00 >> sh) & 0xFF) * (256 - fx) + (int)((c10 >> sh) & 0xFF) * fx;
                    const int b = (int)((c01 >> sh) & 0xFF) * (256 - fx) + (int)((c11 >> sh) & 0xFF) * fx;
                    out |= ((uint32_t)((((t >> 8) * (256 - fy) + (b >> 8) * fy) >> 8)) << sh);
                }
                tc = out;
            } else {
                tc = tpix[opt_wrap(vv >> 16, th, wm_y) * tex->w + opt_wrap(uu >> 16, tw, wm_x)];
            }
            if (!z_test || (uint16_t)zz <= *zb) {
                if (ff < 0) ff = 0; else if (ff > 255) ff = 255;
                const int ia = 255 - ff;
                const int tr = ((tc & 0xFF) * rr + 127) >> 8;
                const int tg = (((tc >> 8) & 0xFF) * gg + 127) >> 8;
                const int tb = (((tc >> 16) & 0xFF) * bb + 127) >> 8;
                // FIX(v9.5): fog blend direction (was inverted, see
                // opt_scan_fog_rgb): out = fog*fog_color + (255-fog)*color
                opt_put(dst,
                    (ia * tr + ff * fcr + 127) >> 8,
                    (ia * tg + ff * fcg + 127) >> 8,
                    (ia * tb + ff * fcb + 127) >> 8);
                if (z_write) *zb = (uint16_t)zz;
            }
            ++dst; ++zb;
            uu += du; vv += dv; ff += df;
            rr += dr; gg += dg; bb += db;
            zz += dz;
        }
        for (int i = 2; i < 10; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- texture only, no shade (combine_tex): skybox with fog OFF.
 *     FIX(v10.2): the SH_MT_TEXTURE branch in create_and_load_new_shader
 *     does NOT check opt_alpha, so combine_tex shaders can carry
 *     DRAW_BLEND / DRAW_BLEND_EDGE (file-select hand, trees, RGBA
 *     overlays). The scanline must therefore resolve the draw mode and
 *     use the TEXEL alpha, exactly like opt_scan_tex_fog (see v9.2):
 *     the previous opaque-only version painted transparent texels
 *     black (square/triangle artifacts around sprites). */
static void opt_scan_tex(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    const int fw16 = (int)(tex->fw * 65536.0f);
    const int fh16 = (int)(tex->fh * 65536.0f);
    const int zoff = (int)z_offset;
    const uint8_t wm_x = tex->wrap_mode_x, wm_y = tex->wrap_mode_y;
    const int tw = tex->wrap_w, th = tex->wrap_h;
    const uint32_t *tpix = (const uint32_t *)(texcache + tex->addr);
    const bool bil = tex->filter;
    const uint32_t dfl = cur_shader->draw_flags;
    const bool do_blend = (dfl & (DRAW_BLEND | DRAW_BLEND_EDGE)) != 0;
    const bool do_edge  = (dfl & DRAW_BLEND_EDGE) != 0;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int uu = (int)(p[4] * w0 * fw16);
        int vv = (int)(p[5] * w0 * fh16);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[6];
        for (int i = 2; i < 6; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int uu1 = (int)(q[4] * w1 * fw16);
        const int vv1 = (int)(q[5] * w1 * fh16);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int du = (uu1 - uu) / cnt;
        const int dv = (vv1 - vv) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            /* OPTIMIZATION (v11): alpha-first early-out (see opt_scan_tex_rgba):
             * texture-edge fullscreen quads (masks) are mostly transparent. */
            uint32_t c00, c10, c01, c11;
            int fx, fy;
            int ta;
            if (bil) {
                const int x0 = uu >> 16, y0 = vv >> 16;
                fx = (uu >> 8) & 0xFF; fy = (vv >> 8) & 0xFF;
                const int xb = opt_wrap(x0, tw, wm_x), yb = opt_wrap(y0, th, wm_y);
                const int xn = opt_wrap(x0 + 1, tw, wm_x), yn = opt_wrap(y0 + 1, th, wm_y);
                c00 = tpix[yb * tex->w + xb];
                c10 = tpix[yb * tex->w + xn];
                c01 = tpix[yn * tex->w + xb];
                c11 = tpix[yn * tex->w + xn];
                const int t = (int)((c00 >> 24) & 0xFF) * (256 - fx) + (int)((c10 >> 24) & 0xFF) * fx;
                const int b = (int)((c01 >> 24) & 0xFF) * (256 - fx) + (int)((c11 >> 24) & 0xFF) * fx;
                ta = (((t >> 8) * (256 - fy) + (b >> 8) * fy) >> 8);
            } else {
                fx = 0; fy = 0;
                c00 = tpix[opt_wrap(vv >> 16, th, wm_y) * tex->w + opt_wrap(uu >> 16, tw, wm_x)];
                c10 = c01 = c11 = c00;
                ta = (int)((c00 >> 24) & 0xFF);
            }
            if (!z_test || (uint16_t)zz <= *zb) {
                if (do_edge && ta <= 0x80) {
                    // below the edge threshold: nothing written at all
                } else if (!do_blend) {
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    opt_put(dst, tr, tg, tb);
                    if (z_write) *zb = (uint16_t)zz;
                } else if (ta == 0) {
                    // blend with 0 alpha: no-op color write, keep z write
                    if (z_write) *zb = (uint16_t)zz;
                } else {
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    if (ta >= 255) {
                        opt_put(dst, tr, tg, tb);
                    } else {
                        int dr_, dg_, db_;
                        opt_get(dst, &dr_, &dg_, &db_);
                        const int ia = 255 - ta;
                        opt_put(dst,
                            (tr * ta + dr_ * ia + 127) >> 8,
                            (tg * ta + dg_ * ia + 127) >> 8,
                            (tb * ta + db_ * ia + 127) >> 8);
                    }
                    if (z_write) *zb = (uint16_t)zz;
                }
            }
            ++dst; ++zb;
            uu += du; vv += dv;
            zz += dz;
        }
        for (int i = 2; i < 6; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- FAST(v12): combine_tex_rgba_texa (id 0x01A00045) - fullscreen
 * blended overlays (level fades) used the GENERIC path: ~30 triangles
 * covering the screen costed ~400 ms per 60 frames. Same as
 * opt_scan_tex but rgb is modulated by the vertex color; alpha comes
 * from the texel only (cc.a == 0xFF in combine_tex_rgba_texa). */
static void opt_scan_tex_rgba_texa(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    const int fw16 = (int)(tex->fw * 65536.0f);
    const int fh16 = (int)(tex->fh * 65536.0f);
    const int zoff = (int)z_offset;
    const uint8_t wm_x = tex->wrap_mode_x, wm_y = tex->wrap_mode_y;
    const int tw = tex->wrap_w, th = tex->wrap_h;
    const uint32_t *tpix = (const uint32_t *)(texcache + tex->addr);
    const bool bil = tex->filter;
    const uint32_t dfl = cur_shader->draw_flags;
    const bool do_blend = (dfl & (DRAW_BLEND | DRAW_BLEND_EDGE)) != 0;
    const bool do_edge  = (dfl & DRAW_BLEND_EDGE) != 0;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int uu = (int)(p[4] * w0 * fw16);
        int vv = (int)(p[5] * w0 * fh16);
        int rr = (int)(p[6] * w0);
        int gg = (int)(p[7] * w0);
        int bb = (int)(p[8] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[9];
        for (int i = 2; i < 9; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int uu1 = (int)(q[4] * w1 * fw16);
        const int vv1 = (int)(q[5] * w1 * fh16);
        const int rr1 = (int)(q[6] * w1);
        const int gg1 = (int)(q[7] * w1);
        const int bb1 = (int)(q[8] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int du = (uu1 - uu) / cnt;
        const int dv = (vv1 - vv) / cnt;
        const int dr = (rr1 - rr) / cnt;
        const int dg = (gg1 - gg) / cnt;
        const int db = (bb1 - bb) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            /* OPTIMIZATION (v11): alpha-first early-out, see opt_scan_tex */
            uint32_t c00, c10, c01, c11;
            int fx, fy;
            int ta;
            if (bil) {
                const int x0 = uu >> 16, y0 = vv >> 16;
                fx = (uu >> 8) & 0xFF; fy = (vv >> 8) & 0xFF;
                const int xb = opt_wrap(x0, tw, wm_x), yb = opt_wrap(y0, th, wm_y);
                const int xn = opt_wrap(x0 + 1, tw, wm_x), yn = opt_wrap(y0 + 1, th, wm_y);
                c00 = tpix[yb * tex->w + xb];
                c10 = tpix[yb * tex->w + xn];
                c01 = tpix[yn * tex->w + xb];
                c11 = tpix[yn * tex->w + xn];
                const int t = (int)((c00 >> 24) & 0xFF) * (256 - fx) + (int)((c10 >> 24) & 0xFF) * fx;
                const int b = (int)((c01 >> 24) & 0xFF) * (256 - fx) + (int)((c11 >> 24) & 0xFF) * fx;
                ta = ((t >> 8) * (256 - fy) + (b >> 8) * fy) >> 8;
            } else {
                fx = 0; fy = 0;
                c00 = tpix[opt_wrap(vv >> 16, th, wm_y) * tex->w + opt_wrap(uu >> 16, tw, wm_x)];
                c10 = c01 = c11 = c00;
                ta = (int)((c00 >> 24) & 0xFF);
            }
            if (!z_test || (uint16_t)zz <= *zb) {
                if (do_edge && ta <= 0x80) {
                    // below the edge threshold: nothing written at all
                } else if (!do_blend) {
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    opt_put(dst, (tr * rr + 127) >> 8, (tg * gg + 127) >> 8, (tb * bb + 127) >> 8);
                    if (z_write) *zb = (uint16_t)zz;
                } else if (ta == 0) {
                    // blend with 0 alpha: no-op color write, keep z write
                    if (z_write) *zb = (uint16_t)zz;
                } else {
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    const int sr = (tr * rr + 127) >> 8;
                    const int sg = (tg * gg + 127) >> 8;
                    const int sb = (tb * bb + 127) >> 8;
                    if (ta >= 255) {
                        opt_put(dst, sr, sg, sb);
                    } else {
                        int dr_, dg_, db_;
                        opt_get(dst, &dr_, &dg_, &db_);
                        const int ia = 255 - ta;
                        opt_put(dst,
                            (sr * ta + dr_ * ia + 127) >> 8,
                            (sg * ta + dg_ * ia + 127) >> 8,
                            (sb * ta + db_ * ia + 127) >> 8);
                    }
                    if (z_write) *zb = (uint16_t)zz;
                }
            }
            ++dst; ++zb;
            uu += du; vv += dv;
            rr += dr; gg += dg; bb += db;
            zz += dz;
        }
        for (int i = 2; i < 9; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- FAST(v12): combine_tex_fog_rgba (id 0x03200045 & co) - fogged
 * alpha-blended textured geometry (water, translucent walls) ran in
 * the GENERIC path: up to 1300 ms per 60 frames. rgb = fog blend of
 * (texel * vertex color), alpha = texel.a * vertex.a (fog is RGB-only,
 * see FIX(v9.2)). */
static void opt_scan_tex_fog_rgba(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    const int fw16 = (int)(tex->fw * 65536.0f);
    const int fh16 = (int)(tex->fh * 65536.0f);
    const int zoff = (int)z_offset;
    const uint8_t wm_x = tex->wrap_mode_x, wm_y = tex->wrap_mode_y;
    const int tw = tex->wrap_w, th = tex->wrap_h;
    const uint32_t *tpix = (const uint32_t *)(texcache + tex->addr);
    const bool bil = tex->filter;
    const int fcr = fog_color.r, fcg = fog_color.g, fcb = fog_color.b;
    const uint32_t dfl = cur_shader->draw_flags;
    const bool do_blend = (dfl & (DRAW_BLEND | DRAW_BLEND_EDGE)) != 0;
    const bool do_edge  = (dfl & DRAW_BLEND_EDGE) != 0;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int uu = (int)(p[4] * w0 * fw16);
        int vv = (int)(p[5] * w0 * fh16);
        int ff = (int)(p[6] * w0);
        int rr = (int)(p[7] * w0);
        int gg = (int)(p[8] * w0);
        int bb = (int)(p[9] * w0);
        int aa = (int)(p[10] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[11];
        for (int i = 2; i < 11; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int uu1 = (int)(q[4] * w1 * fw16);
        const int vv1 = (int)(q[5] * w1 * fh16);
        const int ff1 = (int)(q[6] * w1);
        const int rr1 = (int)(q[7] * w1);
        const int gg1 = (int)(q[8] * w1);
        const int bb1 = (int)(q[9] * w1);
        const int aa1 = (int)(q[10] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int du = (uu1 - uu) / cnt;
        const int dv = (vv1 - vv) / cnt;
        const int df = (ff1 - ff) / cnt;
        const int dr = (rr1 - rr) / cnt;
        const int dg = (gg1 - gg) / cnt;
        const int db = (bb1 - bb) / cnt;
        const int da = (aa1 - aa) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            /* OPTIMIZATION (v11): alpha-first early-out (see
             * opt_scan_tex_rgba): translucent fogged geometry is often
             * mostly transparent. ta = texel alpha modulated by the
             * vertex alpha; fog never touches alpha (FIX(v9.2)). */
            uint32_t c00, c10, c01, c11;
            int fx, fy;
            int ta;
            if (bil) {
                const int x0 = uu >> 16, y0 = vv >> 16;
                fx = (uu >> 8) & 0xFF; fy = (vv >> 8) & 0xFF;
                const int xb = opt_wrap(x0, tw, wm_x), yb = opt_wrap(y0, th, wm_y);
                const int xn = opt_wrap(x0 + 1, tw, wm_x), yn = opt_wrap(y0 + 1, th, wm_y);
                c00 = tpix[yb * tex->w + xb];
                c10 = tpix[yb * tex->w + xn];
                c01 = tpix[yn * tex->w + xb];
                c11 = tpix[yn * tex->w + xn];
                const int t = (int)((c00 >> 24) & 0xFF) * (256 - fx) + (int)((c10 >> 24) & 0xFF) * fx;
                const int b = (int)((c01 >> 24) & 0xFF) * (256 - fx) + (int)((c11 >> 24) & 0xFF) * fx;
                ta = ((((t >> 8) * (256 - fy) + (b >> 8) * fy) >> 8) * aa + 127) >> 8;
            } else {
                fx = 0; fy = 0;
                c00 = tpix[opt_wrap(vv >> 16, th, wm_y) * tex->w + opt_wrap(uu >> 16, tw, wm_x)];
                c10 = c01 = c11 = c00;
                ta = ((int)((c00 >> 24) & 0xFF) * aa + 127) >> 8;
            }
            if (!z_test || (uint16_t)zz <= *zb) {
                if (do_edge && ta <= 0x80) {
                    // below the edge threshold: nothing written at all
                } else if (!do_blend) {
                    if (ff < 0) ff = 0; else if (ff > 255) ff = 255;
                    const int ia = 255 - ff;
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    const int fr = (ia * ((tr * rr + 127) >> 8) + ff * fcr + 127) >> 8;
                    const int fg = (ia * ((tg * gg + 127) >> 8) + ff * fcg + 127) >> 8;
                    const int fb = (ia * ((tb * bb + 127) >> 8) + ff * fcb + 127) >> 8;
                    opt_put(dst, fr, fg, fb);
                    if (z_write) *zb = (uint16_t)zz;
                } else if (ta == 0) {
                    // blend with 0 alpha: the color write is a no-op,
                    // keep only the z write (draw_pixel_blend_zwrite)
                    if (z_write) *zb = (uint16_t)zz;
                } else {
                    if (ff < 0) ff = 0; else if (ff > 255) ff = 255;
                    const int ia = 255 - ff;
                    const int tr_t = ((int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx) >> 8;
                    const int tr_b = ((int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx) >> 8;
                    const int tr = bil ? (tr_t * (256 - fy) + tr_b * fy) >> 8 : (int)(c00 & 0xFF);
                    const int tg_t = ((int)((c00 >> 8) & 0xFF) * (256 - fx) + (int)((c10 >> 8) & 0xFF) * fx) >> 8;
                    const int tg_b = ((int)((c01 >> 8) & 0xFF) * (256 - fx) + (int)((c11 >> 8) & 0xFF) * fx) >> 8;
                    const int tg = bil ? (tg_t * (256 - fy) + tg_b * fy) >> 8 : (int)((c00 >> 8) & 0xFF);
                    const int tb_t = ((int)((c00 >> 16) & 0xFF) * (256 - fx) + (int)((c10 >> 16) & 0xFF) * fx) >> 8;
                    const int tb_b = ((int)((c01 >> 16) & 0xFF) * (256 - fx) + (int)((c11 >> 16) & 0xFF) * fx) >> 8;
                    const int tb = bil ? (tb_t * (256 - fy) + tb_b * fy) >> 8 : (int)((c00 >> 16) & 0xFF);
                    const int fr = (ia * ((tr * rr + 127) >> 8) + ff * fcr + 127) >> 8;
                    const int fg = (ia * ((tg * gg + 127) >> 8) + ff * fcg + 127) >> 8;
                    const int fb = (ia * ((tb * bb + 127) >> 8) + ff * fcb + 127) >> 8;
                    if (ta >= 255) {
                        opt_put(dst, fr, fg, fb);
                    } else {
                        int dr_, dg_, db_;
                        opt_get(dst, &dr_, &dg_, &db_);
                        const int iat = 255 - ta;
                        opt_put(dst,
                            (fr * ta + dr_ * iat + 127) >> 8,
                            (fg * ta + dg_ * iat + 127) >> 8,
                            (fb * ta + db_ * iat + 127) >> 8);
                    }
                    if (z_write) *zb = (uint16_t)zz;
                }
            }
            ++dst; ++zb;
            uu += du; vv += dv; ff += df;
            rr += dr; gg += dg; bb += db; aa += da;
            zz += dz;
        }
        for (int i = 2; i < 11; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- FAST(v12): combine_rgba_rgba (id 0x01081081) - untextured
 * dual-color blend overlays (fade circles) ran in the GENERIC path:
 * ~80 triangles costed ~420 ms per 60 frames. out = ca * cb on all
 * 4 channels (see rgba_modulate). */
static void opt_scan_rgba_rgba(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    (void)tex;
    const int zoff = (int)z_offset;
    const uint32_t dfl = cur_shader->draw_flags;
    const bool do_blend = (dfl & (DRAW_BLEND | DRAW_BLEND_EDGE)) != 0;
    const bool do_edge  = (dfl & DRAW_BLEND_EDGE) != 0;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int r1 = (int)(p[4] * w0);
        int g1 = (int)(p[5] * w0);
        int b1 = (int)(p[6] * w0);
        int a1 = (int)(p[7] * w0);
        int r2 = (int)(p[8] * w0);
        int g2 = (int)(p[9] * w0);
        int b2 = (int)(p[10] * w0);
        int a2 = (int)(p[11] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[12];
        for (int i = 2; i < 12; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int r12 = (int)(q[4] * w1);
        const int g12 = (int)(q[5] * w1);
        const int b12 = (int)(q[6] * w1);
        const int a12 = (int)(q[7] * w1);
        const int r22 = (int)(q[8] * w1);
        const int g22 = (int)(q[9] * w1);
        const int b22 = (int)(q[10] * w1);
        const int a22 = (int)(q[11] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int d1r = (r12 - r1) / cnt;
        const int d1g = (g12 - g1) / cnt;
        const int d1b = (b12 - b1) / cnt;
        const int d1a = (a12 - a1) / cnt;
        const int d2r = (r22 - r2) / cnt;
        const int d2g = (g22 - g2) / cnt;
        const int d2b = (b22 - b2) / cnt;
        const int d2a = (a22 - a2) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            const int ta = (a1 * a2 + 127) >> 8;
            if (!z_test || (uint16_t)zz <= *zb) {
                if (do_edge && ta <= 0x80) {
                    // below the edge threshold: nothing written at all
                } else if (!do_blend) {
                    opt_put(dst, (r1 * r2 + 127) >> 8, (g1 * g2 + 127) >> 8, (b1 * b2 + 127) >> 8);
                    if (z_write) *zb = (uint16_t)zz;
                } else if (ta == 0) {
                    // blend with 0 alpha: no-op color write, keep z write
                    if (z_write) *zb = (uint16_t)zz;
                } else {
                    const int sr = (r1 * r2 + 127) >> 8;
                    const int sg = (g1 * g2 + 127) >> 8;
                    const int sb = (b1 * b2 + 127) >> 8;
                    if (ta >= 255) {
                        opt_put(dst, sr, sg, sb);
                    } else {
                        int dr_, dg_, db_;
                        opt_get(dst, &dr_, &dg_, &db_);
                        const int ia = 255 - ta;
                        opt_put(dst,
                            (sr * ta + dr_ * ia + 127) >> 8,
                            (sg * ta + dg_ * ia + 127) >> 8,
                            (sb * ta + db_ * ia + 127) >> 8);
                    }
                    if (z_write) *zb = (uint16_t)zz;
                }
            }
            ++dst; ++zb;
            r1 += d1r; g1 += d1g; b1 += d1b; a1 += d1a;
            r2 += d2r; g2 += d2g; b2 += d2b; a2 += d2a;
            zz += dz;
        }
        for (int i = 2; i < 12; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- FAST(v12): combine_tex_rgb_rgb (id 0x00000551 & co) - the
 * environment "gradient" shader: texel red channel lerps between two
 * vertex colors. Ran in the GENERIC path (~180 ms per 60 frames in
 * tested scenes). Only the RED texel channel is needed: the bilinear
 * fetch interpolates a single channel. Opaque (alpha == 0xFF from
 * rgba_lerp on two 0xFF alphas). */
static void opt_scan_tex_rgb_rgb(opt_pix_t *dst, uint16_t *zb, int n,
        float *p, const Vector2 *dp, const struct Texture * const tex) {
    const int fw16 = (int)(tex->fw * 65536.0f);
    const int fh16 = (int)(tex->fh * 65536.0f);
    const int zoff = (int)z_offset;
    const uint8_t wm_x = tex->wrap_mode_x, wm_y = tex->wrap_mode_y;
    const int tw = tex->wrap_w, th = tex->wrap_h;
    const uint32_t *tpix = (const uint32_t *)(texcache + tex->addr);
    const bool bil = tex->filter;
    const uint32_t dfl = cur_shader->draw_flags;
    const bool do_blend = (dfl & (DRAW_BLEND | DRAW_BLEND_EDGE)) != 0;
    const bool do_edge  = (dfl & DRAW_BLEND_EDGE) != 0;

    while (n > 0) {
        const int cnt = (n > OPT_STEP) ? OPT_STEP : n;
        const float w0 = 1.0f / p[3];
        int uu = (int)(p[4] * w0 * fw16);
        int vv = (int)(p[5] * w0 * fh16);
        int r1 = (int)(p[6] * w0);
        int g1 = (int)(p[7] * w0);
        int b1 = (int)(p[8] * w0);
        int r2 = (int)(p[9] * w0);
        int g2 = (int)(p[10] * w0);
        int b2 = (int)(p[11] * w0);
        int zz = (int)(p[2] * 65535.0f) + zoff;
        if (zz < 0) zz = 0; else if (zz > 0xFFFF) zz = 0xFFFF;

        float q[12];
        for (int i = 2; i < 12; ++i) q[i] = p[i] + cnt * dp[i].x;
        const float w1 = 1.0f / q[3];
        const int uu1 = (int)(q[4] * w1 * fw16);
        const int vv1 = (int)(q[5] * w1 * fh16);
        const int r12 = (int)(q[6] * w1);
        const int g12 = (int)(q[7] * w1);
        const int b12 = (int)(q[8] * w1);
        const int r22 = (int)(q[9] * w1);
        const int g22 = (int)(q[10] * w1);
        const int b22 = (int)(q[11] * w1);
        int zz1 = (int)(q[2] * 65535.0f) + zoff;
        if (zz1 < 0) zz1 = 0; else if (zz1 > 0xFFFF) zz1 = 0xFFFF;

        const int du = (uu1 - uu) / cnt;
        const int dv = (vv1 - vv) / cnt;
        const int d1r = (r12 - r1) / cnt;
        const int d1g = (g12 - g1) / cnt;
        const int d1b = (b12 - b1) / cnt;
        const int d2r = (r22 - r2) / cnt;
        const int d2g = (g22 - g2) / cnt;
        const int d2b = (b22 - b2) / cnt;
        const int dz = (zz1 - zz) / cnt;

        for (int k = cnt; k; --k) {
            // texel red channel only (see combine_tex_rgb_rgb)
            int t;
            if (bil) {
                const int x0 = uu >> 16, y0 = vv >> 16;
                const int fx = (uu >> 8) & 0xFF, fy = (vv >> 8) & 0xFF;
                const int xb = opt_wrap(x0, tw, wm_x), yb = opt_wrap(y0, th, wm_y);
                const int xn = opt_wrap(x0 + 1, tw, wm_x), yn = opt_wrap(y0 + 1, th, wm_y);
                const uint32_t c00 = tpix[yb * tex->w + xb];
                const uint32_t c10 = tpix[yb * tex->w + xn];
                const uint32_t c01 = tpix[yn * tex->w + xb];
                const uint32_t c11 = tpix[yn * tex->w + xn];
                const int tt = (int)(c00 & 0xFF) * (256 - fx) + (int)(c10 & 0xFF) * fx;
                const int tb_ = (int)(c01 & 0xFF) * (256 - fx) + (int)(c11 & 0xFF) * fx;
                t = ((tt >> 8) * (256 - fy) + (tb_ >> 8) * fy) >> 8;
            } else {
                t = (int)(tpix[opt_wrap(vv >> 16, th, wm_y) * tex->w + opt_wrap(uu >> 16, tw, wm_x)] & 0xFF);
            }
            if (!z_test || (uint16_t)zz <= *zb) {
                // rgba_lerp(cc2, cc1, t): out = cc1 * t + cc2 * (255 - t)
                const int ia = 255 - t;
                const int orr = (r1 * t + r2 * ia + 127) >> 8;
                const int og = (g1 * t + g2 * ia + 127) >> 8;
                const int ob = (b1 * t + b2 * ia + 127) >> 8;
                if (!do_blend) {
                    opt_put(dst, orr, og, ob);
                    if (z_write) *zb = (uint16_t)zz;
                } else if (do_edge) {
                    // unreachable in practice (both alphas 0xFF), kept
                    // for consistency with the other fast paths
                    opt_put(dst, orr, og, ob);
                    if (z_write) *zb = (uint16_t)zz;
                } else {
                    // blend with ta == 0xFF: plain write
                    opt_put(dst, orr, og, ob);
                    if (z_write) *zb = (uint16_t)zz;
                }
            }
            ++dst; ++zb;
            uu += du; vv += dv;
            r1 += d1r; g1 += d1g; b1 += d1b;
            r2 += d2r; g2 += d2g; b2 += d2b;
            zz += dz;
        }
        for (int i = 2; i < 12; ++i) p[i] = q[i];
        n -= cnt;
    }
}

/* --- parameterized triangle walker: same setup math as R_RASTERIZE,
 *     but calls a fused scanline function (no per-pixel indirect calls) --- */
#define R_RASTERIZE_TRI_SEG_FAST(y_a, y_b, NP, FN) \
    register int y = y_a; \
    register int y_end = y_b; \
    register int x, x_end; \
    register int idx; \
    register float dx; \
    while (y < y_end) { \
        x = imax(r_clip.x0, x_a); \
        x_end = imin(r_clip.x1, x_b); \
        dx = 1.f - (x_a - x); \
        for (i = 2; i < NP; ++i) p[i] = p_a[i] + dx * dp[i].x; \
        idx = scr_width * (scr_height - y - 1) + x; \
        if (x_end > x) \
            FN(gfx_output + idx, z_buffer + idx, x_end - x, p, dp, cur_tex[0]); \
        x_a += dxdy_a; \
        x_b += dxdy_b; \
        for (i = 2; i < NP; ++i) p_a[i] += dpdy_a[i]; \
        ++y; \
    }

#define R_RASTERIZE_FAST(tri, NP, FN) \
    const float *v0 = (float *)tri.v0; \
    const float *v1 = (float *)tri.v1; \
    const float *v2 = (float *)tri.v2; \
    const int y0i = imax(r_clip.y0, (int)v0[1]); \
    const int y1i = imax(y0i, (int)v1[1]); \
    const int y2i = imin(r_clip.y1, (int)v2[1]); \
    if ((y0i == y1i && y0i == y2i) || ((int)v0[0] == (int)v1[0] && (int)v0[0] == (int)v2[0])) \
        return; \
    const Vector4 ab = (Vector4) {{ v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], v1[3] - v0[3] }}; \
    const Vector4 ac = (Vector4) {{ v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2], v2[3] - v0[3] }}; \
    const Vector2 bc = (Vector2) {{ v2[0] - v1[0], v2[1] - v1[1] }}; \
    const float denom = 1.f / (ac.x * ab.y - ab.x * ac.y); \
    const float dxdy_ab = ab.x / ab.y; \
    const float dxdy_ac = ac.x / ac.y; \
    const float dxdy_bc = bc.x / bc.y; \
    const bool side = dxdy_ac > dxdy_ab; \
    const float y_pre0 = 1.f - (v0[1] - y0i); \
    float dpdy_a[NP]; \
    float p_a[NP]; \
    float p[NP]; \
    Vector2 dp[NP]; \
    register int i; \
    for (i = 2; i < NP; ++i) { \
        dp[i].x = ((v2[i] - v0[i]) * ab.y - (v1[i] - v0[i]) * ac.y) * denom; \
        dp[i].y = ((v1[i] - v0[i]) * ac.x - (v2[i] - v0[i]) * ab.x) * denom; \
    } \
    if (!side) { \
        const float dxdy_a = dxdy_ac; \
        float x_a = v0[0] + y_pre0 * dxdy_a; \
        for (i = 2; i < NP; ++i) { \
            dpdy_a[i] = dxdy_ac * dp[i].x + dp[i].y; \
            p_a[i] = v0[i] + y_pre0 * dpdy_a[i]; \
        } \
        if (y0i < y1i) { \
            const float dxdy_b = dxdy_ab; \
            float x_b = v0[0] + y_pre0 * dxdy_ab; \
            R_RASTERIZE_TRI_SEG_FAST(y0i, y1i, NP, FN); \
        } \
        if (y1i < y2i) { \
            const float dxdy_b = dxdy_bc; \
            const float y_pre1 = 1.f - (v1[1] - y1i); \
            float x_b = v1[0] + y_pre1 * dxdy_bc; \
            R_RASTERIZE_TRI_SEG_FAST(y1i, y2i, NP, FN); \
        } \
    } else { \
        const float dxdy_b = dxdy_ac; \
        float x_b = v0[0] + y_pre0 * dxdy_ac; \
        if (y0i < y1i) { \
            const float dxdy_a = dxdy_ab; \
            float x_a = v0[0] + y_pre0 * dxdy_a; \
            for (i = 2; i < NP; ++i) { \
                dpdy_a[i] = dxdy_ab * dp[i].x + dp[i].y; \
                p_a[i] = v0[i] + y_pre0 * dpdy_a[i]; \
            } \
            R_RASTERIZE_TRI_SEG_FAST(y0i, y1i, NP, FN); \
        } \
        if (y1i < y2i) { \
            const float y_pre1 = 1.f - (v1[1] - y1i); \
            const float dxdy_a = dxdy_bc; \
            float x_a = v1[0] + y_pre1 * dxdy_a; \
            for (i = 2; i < NP; ++i) { \
                dpdy_a[i] = dxdy_bc * dp[i].x + dp[i].y; \
                p_a[i] = v1[i] + y_pre1 * dpdy_a[i]; \
            } \
            R_RASTERIZE_TRI_SEG_FAST(y1i, y2i, NP, FN); \
        } \
    }

static void rast_fn_fast_tex_rgb(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 9, opt_scan_tex_rgb);
}
static void rast_fn_fast_tex_rgba(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 10, opt_scan_tex_rgba);
}
static void rast_fn_fast_rgba(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 8, opt_scan_rgba);
}
static void rast_fn_fast_rgb(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 7, opt_scan_rgb);
}
static void rast_fn_fast_tex_fog(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 7, opt_scan_tex_fog);
}
static void rast_fn_fast_fog_rgb(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 8, opt_scan_fog_rgb);
}
static void rast_fn_fast_tex_fog_rgb(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 10, opt_scan_tex_fog_rgb);
}
static void rast_fn_fast_tex(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 6, opt_scan_tex);
}
// FAST(v12): previously-GENERIC shader combinations (see opt_scan_*)
static void rast_fn_fast_tex_rgba_texa(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 10, opt_scan_tex_rgba_texa);
}
static void rast_fn_fast_tex_fog_rgba(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 11, opt_scan_tex_fog_rgba);
}
static void rast_fn_fast_rgba_rgba(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 12, opt_scan_rgba_rgba);
}
static void rast_fn_fast_tex_rgb_rgb(const struct Tri tri) {
    R_RASTERIZE_FAST(tri, 12, opt_scan_tex_rgb_rgb);
}

static inline void pop_triangle(const float *buf, const int stride) {
    Vector4 *v0 = (Vector4 *)buf;
    Vector4 *v1 = (Vector4 *)(buf + stride);
    Vector4 *v2 = (Vector4 *)(buf + (stride << 1));
    Vector4 *vt;

    // the vertices come to us in clip space, but already divided by w, still gotta transform
    viewport_transform(v0);
    viewport_transform(v1);
    viewport_transform(v2);

    // sort in Y order
    if (v0->y > v1->y) { vt = v0; v0 = v1; v1 = vt; }
    if (v0->y > v2->y) { vt = v0; v0 = v2; v2 = vt; }
    if (v1->y > v2->y) { vt = v1; v1 = v2; v2 = vt; }

    const struct Tri out = (struct Tri) { (float *)v0, (float *)v1, (float *)v2 };
    cur_shader->rast(out);
}

static inline void depth_clear(void) {
    memset(z_buffer, 0xFF, scr_size << 1);
}

static inline void color_clear(void) {
    // FIX: was hardcoded scr_size << 2 (4 bytes/pixel), which overflowed
    // the RGB565 framebuffer by 2x under CONVERT. sizeof(*) is correct
    // in both builds.
    memset(gfx_output, 0x00, scr_size * sizeof(*gfx_output));
}

/* FIXME: ztrick fucks with sky blending
static inline void depth_swap(void) {
    ++z_frame;
    if (z_frame & 1) {
        r_view.zn = 0.f;
        r_view.zf = 0.4999f;
        z_reverse = false;
    } else {
        r_view.zn = 1.f;
        r_view.zf = 0.5f;
        z_reverse = true;
    }
    r_view.hz = (r_view.zf - r_view.zn) * 0.5f;
    r_view.cz = (r_view.zn + r_view.zf) * 0.5f;
}
*/

/* interface */

static bool gfx_soft_z_is_from_0_to_1(void) {
    return true;
}

static void gfx_soft_unload_shader(struct ShaderProgram *old_prg) {
    if (cur_shader && (cur_shader == old_prg || !old_prg))
        cur_shader = NULL;
}

static void gfx_soft_load_shader(struct ShaderProgram *new_prg) {
    cur_shader = new_prg;
}

static struct ShaderProgram *gfx_soft_create_and_load_new_shader(uint32_t shader_id) {
    static const rast_fn_t rast_funcs[] = {
        NULL,
        NULL,
        GET_RAST_FUNC(6),
        GET_RAST_FUNC(7),
        GET_RAST_FUNC(8),
        GET_RAST_FUNC(9),
        GET_RAST_FUNC(10),
        GET_RAST_FUNC(11),
        GET_RAST_FUNC(12),
        GET_RAST_FUNC(13),
        GET_RAST_FUNC(14),
    };

    struct CCFeatures ccf;
    gfx_cc_get_features(shader_id, &ccf);

    struct ShaderProgram *prg = &shader_program_pool[shader_program_pool_size++];

    prg->shader_id = shader_id;
    prg->cc = ccf;

    // FIX(v9.3): the hardcoded combiner ids below were captured with fog
    // off. Enabling fog ORs SHADER_OPT_FOG into the id, so those checks
    // silently stopped matching -> decal geometry got the generic fog
    // combiner (texel modulated by gray vertex color). When fog is on,
    // compare with the fog bit masked off so decal shaders keep their
    // dedicated combiners; with fog off the id is compared raw, exactly
    // as before.
    const uint32_t base_id = ccf.opt_fog ? (shader_id & ~(uint32_t)SHADER_OPT_FOG) : shader_id;

    int num_props = 0;

    if (ccf.opt_fog) num_props++; // software renderer only gets fog intensity

    num_props += ccf.num_inputs * (ccf.opt_alpha ? 4 : 3);
    num_props += ccf.used_textures[0] * 2;

    if (ccf.used_textures[0] && ccf.used_textures[1]) {
        prg->mix = SH_MT_TEXTURE_TEXTURE;
        prg->combine = combine_tex_tex_rgba; // only one such known shader
    } else if (ccf.used_textures[0] && ccf.num_inputs) {
        prg->mix = SH_MT_TEXTURE_COLOR;
        if (ccf.num_inputs > 1)
            prg->combine = combine_tex_rgb_rgb; // only one such known shader
        else if (base_id == 0x0000038D || base_id == 0x01200A00 || base_id == 0x01045A00 || base_id == 0x0120038D)
            if (ccf.opt_alpha) {
                bool alpha_uses_texel = false;
                for (int k = 0; k < 4; k++)
                    if (ccf.c[1][k] == SHADER_TEXEL0 || ccf.c[1][k] == SHADER_TEXEL0A)
                        alpha_uses_texel = true;
                prg->combine = ccf.opt_fog
                    ? (alpha_uses_texel ? combine_tex_fog_rgba_decal_texa : combine_tex_fog_rgba_decal)
                    : (alpha_uses_texel ? combine_tex_rgba_decal_texa : combine_tex_rgba_decal);
            } else
                prg->combine = ccf.opt_fog ? combine_tex_fog_rgb_decal : combine_tex_rgb_decal;
        else if (ccf.opt_fog)
            prg->combine = ccf.opt_alpha ? combine_tex_fog_rgba : combine_tex_fog_rgb;
        else if (ccf.opt_alpha)
            prg->combine = base_id == 0x01A00045 ? combine_tex_rgba_texa : combine_tex_rgba;
        else
            prg->combine = combine_tex_rgb;
    } else if (ccf.used_textures[0]) {
        prg->mix = SH_MT_TEXTURE;
        prg->combine = ccf.opt_fog ? combine_tex_fog : combine_tex;
    } else if (ccf.num_inputs > 1) {
        prg->mix = SH_MT_COLOR_COLOR;
        prg->combine = combine_rgba_rgba; // only one such known shader
    } else if (ccf.num_inputs) {
        prg->mix = SH_MT_COLOR;
        if (ccf.opt_fog)
            prg->combine = ccf.opt_alpha ? combine_fog_rgba : combine_fog_rgb;
        else
            prg->combine = ccf.opt_alpha ? combine_rgba : combine_rgb;
    }

    if (ccf.opt_alpha) {
        if (ccf.opt_texture_edge)
            prg->draw_flags = DRAW_BLEND_EDGE;
        else
            prg->draw_flags = DRAW_BLEND;
    } else {
        prg->draw_flags = 0;
    }

    prg->num_props = num_props;
    // pick rasterizer that interps the amount of float properties this shader requires
    prg->rast = rast_funcs[num_props];

    // OPTIMIZATION: route the dominant shaders to the fused integer-only
    // fast paths (no per-pixel division, no indirect calls). Together they
    // cover the vast majority of rendered pixels in SM64. Works in both
    // 32bpp and CONVERT (RGB565) builds. v9: fog variants get their own
    // fused paths too. Alpha fog variants keep the generic path (rare).
    // NOTE(v9): num_props here is the index into rast_funcs, i.e. it does
    // NOT include the 4 vertex attributes (x, y, z, w). The v8 constants
    // were written with the totals (9/10/8/7) and therefore never matched:
    // the fast paths were dead code. Correct indices: tex_rgb=5, tex_rgba=6,
    // rgba=4, rgb=3, tex_fog_rgb=6, fog_rgb=4, tex_fog=3.
    if (num_props == 5 && prg->combine == combine_tex_rgb)
        prg->rast = rast_fn_fast_tex_rgb;
    else if (num_props == 6 && prg->combine == combine_tex_rgba)
        prg->rast = rast_fn_fast_tex_rgba;
    else if (num_props == 4 && prg->combine == combine_rgba)
        prg->rast = rast_fn_fast_rgba;
    else if (num_props == 3 && prg->combine == combine_rgb)
        prg->rast = rast_fn_fast_rgb;
    else if (num_props == 6 && prg->combine == combine_tex_fog_rgb)
        prg->rast = rast_fn_fast_tex_fog_rgb;
    else if (num_props == 4 && prg->combine == combine_fog_rgb)
        prg->rast = rast_fn_fast_fog_rgb;
    else if (num_props == 3 && prg->combine == combine_tex_fog)
        prg->rast = rast_fn_fast_tex_fog;
    else if (num_props == 2 && prg->combine == combine_tex)
        prg->rast = rast_fn_fast_tex;
    // FAST(v12): the shader ids below were measured running the GENERIC
    // rasterizer at a heavy cost (per-shader timing, 60-frame windows):
    //   0x01A00045 combine_tex_rgba_texa   ~30 tris  ~400 ms (fades)
    //   0x03200045 combine_tex_fog_rgba    ~12k tris ~1300 ms (fog+alpha)
    //   0x01081081 combine_rgba_rgba        ~80 tris  ~420 ms (overlays)
    //   0x00000551 combine_tex_rgb_rgb    ~14k tris  ~180 ms (gradients)
    else if (num_props == 6 && prg->combine == combine_tex_rgba_texa)
        prg->rast = rast_fn_fast_tex_rgba_texa;
    else if (num_props == 7 && prg->combine == combine_tex_fog_rgba)
        prg->rast = rast_fn_fast_tex_fog_rgba;
    else if (num_props == 8 && prg->combine == combine_rgba_rgba)
        prg->rast = rast_fn_fast_rgba_rgba;
    else if (num_props == 8 && prg->combine == combine_tex_rgb_rgb)
        prg->rast = rast_fn_fast_tex_rgb_rgb;

    gfx_soft_load_shader(prg);

    return prg;
}

static struct ShaderProgram *gfx_soft_lookup_shader(uint32_t shader_id) {
    for (size_t i = 0; i < shader_program_pool_size; i++)
        if (shader_program_pool[i].shader_id == shader_id)
            return &shader_program_pool[i];
    return NULL;
}

static void gfx_soft_shader_get_info(struct ShaderProgram *prg, uint8_t *num_inputs, bool used_textures[2]) {
    *num_inputs = prg->cc.num_inputs;
    used_textures[0] = prg->cc.used_textures[0];
    used_textures[1] = prg->cc.used_textures[1];
}

static uint32_t gfx_soft_new_texture(void) {
    const uint32_t id = tex_num++;

    if (tex_num > MAX_TEXTURES) {
        printf("gfx_soft: ran out of texture slots\n");
        abort();
    }

    tex_hdr[id].sample = tex_sample_nearest_rr;
    tex_hdr[id].wrap_mode_x = WRAP_REPEAT;
    tex_hdr[id].wrap_mode_y = WRAP_REPEAT;

    return id;
}

static void gfx_soft_select_texture(int tile, uint32_t texture_id) {
    cur_tex[tile] = tex_hdr + texture_id;
    cur_tmu = tile;
}

static uint32_t tex_cache_alloc(const uint32_t w, const uint32_t h) {
    const uint32_t size = w * h * 4;

    if (texcache_addr + size > texcache_size) {
        texcache_size += TEXCACHE_STEP + size;
        texcache_size = ALIGN(texcache_size, TEXCACHE_STEP);
        texcache = realloc(texcache, texcache_size);
        if (!texcache) {
            printf("gfx_soft: could not alloc %u bytes for texture cache\n", texcache_size);
            abort();
        }
    }

    uint32_t ret = texcache_addr;
    texcache_addr += size;
    return ret;
}

static void gfx_soft_upload_texture(const uint8_t *rgba32_buf, int width, int height) {
    uint32_t addr = tex_cache_alloc(width, height);
    memcpy(texcache + addr, rgba32_buf, width * height * 4);
    struct Texture *tex = cur_tex[cur_tmu];
    tex->addr = addr;
    tex->w = width;
    tex->h = height;
    tex->wrap_w = width - 1;
    tex->wrap_h = height - 1;
    tex->fw = (float)tex->w;
    tex->fh = (float)tex->h;
}

static inline int gfx_cm_to_local(uint32_t val) {
    if (val & G_TX_CLAMP) return WRAP_CLAMP;
    return (val & G_TX_MIRROR) ? WRAP_MIRROR : WRAP_REPEAT;
}

static void gfx_soft_set_sampler_parameters(int tile, bool linear_filter, uint32_t cms, uint32_t cmt) {
    static const sample_fn_t samplers[] = {
        tex_sample_nearest_rr, // 0000
        tex_sample_nearest_rc, // 0001
        tex_sample_nearest_rm, // 0010
        NULL,
        tex_sample_nearest_cr, // 0100
        tex_sample_nearest_cc, // 0101
        tex_sample_nearest_cm, // 0110
        NULL,
        tex_sample_nearest_mr, // 1000
        tex_sample_nearest_mc, // 1001
        tex_sample_nearest_mm, // 1010
    };

    cms = gfx_cm_to_local(cms) << 2;
    cmt = gfx_cm_to_local(cmt);

    cur_tex[tile]->filter = linear_filter;
    cur_tex[tile]->sample = samplers[cms | cmt];
    // OPTIMIZATION: resolve wrap mode per axis once so the fast path
    // can wrap with 2 integer ops instead of an indirect call
    cur_tex[tile]->wrap_mode_x = (uint8_t)(cms >> 2);
    cur_tex[tile]->wrap_mode_y = (uint8_t)cmt;
}

static void gfx_soft_set_depth_test(bool depth_test) {
    z_test = depth_test;
}

static void gfx_soft_set_depth_mask(bool z_upd) {
    z_write = z_upd;
}

static void gfx_soft_set_zmode_decal(bool zmode_decal) {
    z_offset = zmode_decal ? -32.f : 0.f;
}

static void gfx_soft_set_viewport(int x, int y, int width, int height) {
    r_view.x = x;
    r_view.y = y;
    r_view.w = width;
    r_view.h = height;
    r_view.hw = width >> 1;
    r_view.hh = height >> 1;
    r_view.cx = x + r_view.hw;
    r_view.cy = y + r_view.hh;
}

static void gfx_soft_set_scissor(int x, int y, int width, int height) {
    // Small performance hack for the FunKey S:
    // keep the logic of the graphics to be 4:3 so the HUD isn't messed up
    // but crop here to process less pixels and get better performance
    const int targetx = (gfx_current_dimensions.width - gfx_current_dimensions.height) >> 1;
    const int targetw = gfx_current_dimensions.height; //+1;
    if (x<targetx) x=targetx;
    if (width>targetw) width=targetw;
    r_clip.x0 = x;
    r_clip.y0 = y;
    r_clip.x1 = x + width;
    r_clip.y1 = y + height;
}

static void gfx_soft_set_use_alpha(bool use_alpha) {
    do_blend = use_alpha;
}

static void gfx_soft_set_fog_color(const uint8_t *rgb) {
    fog_color.r = rgb[0];
    fog_color.g = rgb[1];
    fog_color.b = rgb[2];
    fog_color.a = 0xFF;
}

static inline void gfx_soft_pick_draw_func(void) {
    static const draw_fn_t draw_funcs[] = {
        draw_pixel,
        draw_pixel_zwrite,
        draw_pixel_blend,
        draw_pixel_blend_zwrite,
        draw_pixel_blend_edge,
        draw_pixel_blend_edge_zwrite,
    };
    draw_fn = draw_funcs[cur_shader->draw_flags | z_write];
}

static void gfx_soft_draw_triangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
    gfx_soft_pick_draw_func();
    const size_t num_verts = 3 * buf_vbo_num_tris;
    const size_t stride = buf_vbo_len / num_verts;
    for (size_t i = 0; i < num_verts * stride; i += 3 * stride)
        pop_triangle(buf_vbo + i, stride);
}

static void gfx_soft_fill_rect(int x0, int y0, int x1, int y1, const uint8_t *rgba) {
    // HACK: these are mainly used just to clear the screen and draw simple rects, so we ignore drawmode stuff and Z
    x0 = imax(0, x0);
    y0 = imax(0, y0);
    x1 = imin(scr_width, x1);
    y1 = imin(scr_height, y1);
#ifndef CONVERT
    // FIX(CONVERT): the old code unconditionally used uint32_t here, which
    // under CONVERT (uint16_t RGB565 framebuffer) wrote 2x the buffer size
    // on every fill/clear -> heap corruption, the startup crash.
    register const uint32_t color = *(uint32_t *)rgba;
    register uint32_t *base = gfx_output + y0 * scr_width + x0;
    register uint32_t *p;
    register int x, y;
    for (y = y0; y < y1; ++y, base += scr_width) {
        p = base;
        for (x = x0; x < x1; ++x, ++p)
            *p = color;
    }
#else
    // RGB565: pack r,g,b from the fill color (rgba[0..2] = r,g,b)
    register const uint16_t color = opt_pack565(rgba[0], rgba[1], rgba[2]);
    register uint16_t *base = gfx_output + y0 * scr_width + x0;
    register uint16_t *p;
    register int x, y;
    for (y = y0; y < y1; ++y, base += scr_width) {
        p = base;
        for (x = x0; x < x1; ++x, ++p)
            *p = color;
    }
#endif
}

static inline void gfx_soft_tex_rect_replace(int x0, int y0, int x1, int y1, const float u0, const float v0, const float dudx, const float dvdy) {
    register int base = y0 * scr_width + x0;
    register int idx;
    register int x, y;
    float u;
    float v = v0;
    for (y = y0; y < y1; ++y, base += scr_width, v += dvdy) {
        idx = base;
        u = u0;
        for (x = x0; x < x1; ++x, ++idx, u += dudx)
            draw_fn(idx, 0, cur_tex[0]->sample(cur_tex[0], u, v));
    }
}

static inline void gfx_soft_tex_rect_modulate(int x0, int y0, int x1, int y1, const float u0, const float v0, const float dudx, const float dvdy, const Color4 rgba) {
    register int base = y0 * scr_width + x0;
    register int idx;
    register int x, y;
    float u;
    float v = v0;
    for (y = y0; y < y1; ++y, base += scr_width, v += dvdy) {
        idx = base;
        u = u0;
        for (x = x0; x < x1; ++x, ++idx, u += dudx)
            draw_fn(idx, 0, rgba_modulate(cur_tex[0]->sample(cur_tex[0], u, v), rgba));
    }
}

static void gfx_soft_tex_rect(int x0, int y0, int x1, int y1, const float u0, const float v0, const float dudx, const float dvdy, const uint8_t *rgba) {
    x0 = imax(0, x0);
    y0 = imax(0, y0);
    x1 = imin(scr_width, x1);
    y1 = imin(scr_height, y1);
    gfx_soft_pick_draw_func();
    if (cur_shader->cc.num_inputs)
        gfx_soft_tex_rect_modulate(x0, y0, x1, y1, u0, v0, dudx, dvdy, *(Color4 *)rgba);
    else
        gfx_soft_tex_rect_replace(x0, y0, x1, y1, u0, v0, dudx, dvdy);
}

static void gfx_soft_prepare_tables(void) {
    // OPTIMIZATION: tables removed (they polluted L1 caches).
    // Kept as a no-op to preserve the init call sequence.
}

static void gfx_soft_set_resolution(const int width, const int height) {
    if (z_buffer) free(z_buffer);
#ifndef SDL_SURFACE
    if (gfx_output) free(gfx_output);
#endif
    scr_width = width;
    scr_height = height;
    scr_size = scr_width * scr_height;

    z_buffer = calloc(scr_width * scr_height, sizeof(int16_t));
    if (!z_buffer) {
        printf("gfx_soft: could not alloc zbuffer for %dx%d\n", scr_width, scr_height);
        abort();
    }

#ifndef SDL_SURFACE
#ifdef CONVERT
    gfx_output = calloc(scr_width * scr_height, sizeof(uint16_t));
#else
    gfx_output = calloc(scr_width * scr_height, sizeof(uint32_t));
#endif
    if (!gfx_output) {
        printf("gfx_soft: could not alloc color buffer for %dx%d\n", scr_width, scr_height);
        abort();
    }
#endif

    depth_clear();
}

static void gfx_soft_init(void) {
    texcache = calloc(1, TEXCACHE_STEP); // this will be realloc'd as needed
    texcache_size = TEXCACHE_STEP;
    texcache_addr = 0;
    if (!texcache) {
        printf("gfx_soft: could not alloc %u bytes for texture cache\n", TEXCACHE_STEP);
        abort();
    }

    z_test = true;
    z_write = true;
    do_blend = false;
    do_clip = false;

    gfx_soft_prepare_tables();

    gfx_soft_set_resolution(gfx_current_dimensions.width, gfx_current_dimensions.height);
}

static void gfx_soft_start_frame(void) {
    // depth_swap(); // FIXME: ztrick
    depth_clear();
}

static void gfx_soft_shutdown(void) {
    free(z_buffer);
    free(texcache);
}

static void gfx_soft_on_resize(void) {
    gfx_soft_set_resolution(gfx_current_dimensions.width, gfx_current_dimensions.height);
}

static void gfx_soft_end_frame(void) {
}

static void gfx_soft_finish_render(void) {
}

struct GfxRenderingAPI gfx_soft_api = {
    gfx_soft_z_is_from_0_to_1,
    gfx_soft_unload_shader,
    gfx_soft_load_shader,
    gfx_soft_create_and_load_new_shader,
    gfx_soft_lookup_shader,
    gfx_soft_shader_get_info,
    gfx_soft_new_texture,
    gfx_soft_select_texture,
    gfx_soft_upload_texture,
    gfx_soft_set_sampler_parameters,
    gfx_soft_set_depth_test,
    gfx_soft_set_depth_mask,
    gfx_soft_set_zmode_decal,
    gfx_soft_set_viewport,
    gfx_soft_set_scissor,
    gfx_soft_set_use_alpha,
    gfx_soft_draw_triangles,
    gfx_soft_init,
    gfx_soft_on_resize,
    gfx_soft_start_frame,
    gfx_soft_end_frame,
    gfx_soft_finish_render,
    gfx_soft_fill_rect,
    gfx_soft_tex_rect,
    gfx_soft_set_fog_color,
    gfx_soft_shutdown,
};

#endif // ENABLE_OPENGL_LEGACY
