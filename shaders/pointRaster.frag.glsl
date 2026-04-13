#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"

layout ( location = 0 ) in flat uint index;

layout ( location = 0 ) out vec4 outColor;

void main () {
	/*
	// analytic solution for sphere mask/height via pythagoras
	vec2 sampleLocation = gl_PointCoord.xy;
	vec2 centered = sampleLocation * 2.0f - vec2( 1.0f );
	float radiusSquared = dot( centered, centered );
	if ( radiusSquared > 1.0f ) discard;
	*/

	outColor = vec4( 1.0f, 0.0f, 0.0f, 1.0f );
}