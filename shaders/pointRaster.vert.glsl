#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"
#include "random.h"

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

layout ( location = 0 ) out flat uint index;
layout ( location = 1 ) out flat float radius;

void main () {
	vec2 center = ( 2.0f * points[ gl_VertexIndex ].position.xy / GlobalData.accumulatorResolution.xy ) - vec2( 1.0f );
	center.x *= GlobalData.accumulatorResolution.x / GlobalData.accumulatorResolution.y;

	// outputs
	// radius = gl_PointSize = ( PushConstants.pointScale == 1.0f ) ? 1.0f : ( points[ gl_VertexIndex ].mass );
	radius = gl_PointSize = ( PushConstants.pointScale == 1.0f ) ? 1.0f : 5.0f;
	index = gl_VertexIndex;

	// writing the point locations
	gl_Position = vec4( center, 0.0f, 1.0f );
}