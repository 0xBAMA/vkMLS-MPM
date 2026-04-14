#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

layout ( location = 0 ) in flat uint index;

layout ( location = 0 ) out vec4 outColor;

struct point {
	vec2 position;
	vec2 velocity;

	mat2 C;
	mat2 Fs;

	float mass;
	float v0;

	vec2 pad;
};

layout( set = 0, binding = 1, std430 ) readonly buffer pointBuffer {
	point points[];
};

void main () {
	/*
	// analytic solution for sphere mask/height via pythagoras
	vec2 sampleLocation = gl_PointCoord.xy;
	vec2 centered = sampleLocation * 2.0f - vec2( 1.0f );
	float radiusSquared = dot( centered, centered );
	if ( radiusSquared > 1.0f ) discard;
	*/

//	outColor = vec4( 1.0f, 0.0f, 0.0f, 1.0f );
	outColor = vec4( 0.1f + abs( points[ index ].velocity / 10.0f ), 0.1f, 1.0f );
}