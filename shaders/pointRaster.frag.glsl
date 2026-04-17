#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

layout ( location = 0 ) in flat uint index;
layout ( location = 1 ) in flat float radius;

layout ( location = 0 ) out vec4 outColor;

struct point {
	vec2 position;
	vec2 velocity;

	mat2 C;
	mat2 Fs;

	float mass;
	float v0;

	int particleType;
	float pad;
};

layout( set = 0, binding = 1, std430 ) readonly buffer pointBuffer {
	point points[];
};

// Define saturation macro, if not already user-defined
#ifndef saturate
#define saturate(v) clamp(v, 0, 1)
#endif

// Constants
const float HCV_EPSILON = 1e-10;
const float HSL_EPSILON = 1e-10;
const float HCY_EPSILON = 1e-10;

const float SRGB_GAMMA = 1.0 / 2.2;
const float SRGB_INVERSE_GAMMA = 2.2;
const float SRGB_ALPHA = 0.055;

// Used to convert from linear RGB to XYZ space
const mat3 RGB_2_XYZ = ( mat3(
	0.4124564, 0.2126729, 0.0193339,
	0.3575761, 0.7151522, 0.1191920,
	0.1804375, 0.0721750, 0.9503041
) );

// Used to convert from XYZ to linear RGB space
const mat3 XYZ_2_RGB = ( mat3(
	3.2404542, -0.9692660, 0.0556434,
	-1.5371385, 1.8760108, -0.2040259,
	-0.4985314, 0.0415560, 1.0572252
) );

const vec3 LUMA_COEFFS = vec3(0.2126, 0.7152, 0.0722);

// Returns the luminance of a !! linear !! rgb color
float get_luminance ( vec3 rgb ) {
	return dot(LUMA_COEFFS, rgb);
}

// Converts a linear rgb color to a srgb color (approximated, but fast)
vec3 rgb_to_srgb_approx ( vec3 rgb ) {
	return pow(rgb, vec3(SRGB_GAMMA));
}

// Converts a srgb color to a rgb color (approximated, but fast)
vec3 srgb_to_rgb_approx ( vec3 srgb ) {
	return pow(srgb, vec3(SRGB_INVERSE_GAMMA));
}

// Converts a single linear channel to srgb
float linear_to_srgb ( float channel ) {
	if (channel <= 0.0031308)
	return 12.92 * channel;
	else
	return (1.0 + SRGB_ALPHA) * pow(channel, 1.0 / 2.4) - SRGB_ALPHA;
}

// Converts a single srgb channel to rgb
float srgb_to_linear ( float channel ) {
	if (channel <= 0.04045)
	return channel / 12.92;
	else
	return pow((channel + SRGB_ALPHA) / (1.0 + SRGB_ALPHA), 2.4);
}

// Converts a linear rgb color to a srgb color (exact, not approximated)
vec3 rgb_to_srgb ( vec3 rgb ) {
	return vec3(
	linear_to_srgb(rgb.r),
	linear_to_srgb(rgb.g),
	linear_to_srgb(rgb.b));
}

// Converts a srgb color to a linear rgb color (exact, not approximated)
vec3 srgb_to_rgb ( vec3 srgb ) {
	return vec3(
	srgb_to_linear(srgb.r),
	srgb_to_linear(srgb.g),
	srgb_to_linear(srgb.b));
}

// Converts a color from linear RGB to XYZ space
vec3 rgb_to_xyz ( vec3 rgb ) {
	return RGB_2_XYZ * rgb;
}

// Converts a color from XYZ to linear RGB space
vec3 xyz_to_rgb ( vec3 xyz ) {
	return XYZ_2_RGB * xyz;
}

// Converts a color from XYZ to xyY space (Y is luminosity)
vec3 xyz_to_xyY ( vec3 xyz ) {
	float Y = xyz.y;
	float x = xyz.x / (xyz.x + xyz.y + xyz.z);
	float y = xyz.y / (xyz.x + xyz.y + xyz.z);
	return vec3(x, y, Y);
}

// Converts a color from xyY space to XYZ space
vec3 xyY_to_xyz ( vec3 xyY ) {
	float Y = xyY.z;
	float x = Y * xyY.x / xyY.y;
	float z = Y * (1.0 - xyY.x - xyY.y) / xyY.y;
	return vec3(x, Y, z);
}

// Converts a color from linear RGB to xyY space
vec3 rgb_to_xyY ( vec3 rgb ) {
	vec3 xyz = rgb_to_xyz(rgb);
	return xyz_to_xyY(xyz);
}

// Converts a color from xyY space to linear RGB
vec3 xyY_to_rgb ( vec3 xyY ) {
	vec3 xyz = xyY_to_xyz(xyY);
	return xyz_to_rgb(xyz);
}

mat2 Rotate2D ( in float a ) {
	float c = cos( a ), s = sin( a );
	return mat2( c, s, -s, c );
}

#include "random.h"

void main () {
//	seed = PushConstants.wangSeed + 42069 * index;

	// analytic solution for sphere mask/height via pythagoras
	if ( radius != 1.0f ) {
		// vec2 sampleLocation = gl_PointCoord.xy + 0.5f * CircleOffset() / radius;
		vec2 sampleLocation = gl_PointCoord.xy;
		vec2 centered = sampleLocation * 2.0f - vec2( 1.0f );
		float radiusSquared = dot( centered, centered );
		 if ( radiusSquared > 1.0f ) discard;
	}

	if ( points[ index ].particleType == 0 ) { // neo-hookean
		outColor = vec4( 0.1f + abs( points[ index ].velocity / 10.0f ), 0.1f, 1.0f );
	} else if ( points[ index ].particleType == 1 ) { // fluid
		outColor = vec4( mix( xyY_to_rgb( vec3( Rotate2D( 0.3f + 0.1f * sin( atan( points[ index ].velocity.x, points[ index ].velocity.y ) ) ) * vec2( 0.3f, 0.1f ), 0.1f * length( points[ index ].velocity ) ) ), vec3( 0.1f * points[ index ].mass ), 0.5f ), 1.0f );
	}
}